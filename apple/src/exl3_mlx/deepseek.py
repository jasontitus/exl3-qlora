"""Experimental text-only DeepSeek V4.1 Flash trainer model.

Loads the mtplx streaming repack: MXFP8/BF16 trunk, real EXL3 routed experts,
and sparse MXFP8 Engram banks. All 40 backbone layers participate in text loss.
No vision, MTP, decode cache, or PEFT export is claimed.
"""

import json
import platform
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from .activation_config import activation_description, validate_activation_mode
from .attention import FrozenIndexer, ShortAttention, empty_attention_state
from .context import MAX_CONTEXT_TOKENS
from .engram import Engram, EngramBank
from .hyperconnections import HyperConnection, expand_streams, hc_post, hc_pre, rms_norm
from .moe import DeepSeekExpert, DeepSeekGate, DeepSeekMoE
from .native import LoRALinear, NativeLinear
from .prefix_cache import FrozenPrefixCache, frozen_structure_signature
from .prefix_disk import FrozenPrefixDisk, fingerprint, hash_payloads, validate_disk_options
from .recompute import recompute
from .storage import ExpertBank, TensorArchive
from .streaming import MatrixStore


class TrainingBlock(nn.Module):
    def __init__(
        self,
        attn,
        ffn,
        attn_hc,
        ffn_hc,
        attn_norm,
        ffn_norm,
        engram=None,
        *,
        checkpoint_attention=False,
    ):
        super().__init__()
        self.attn, self.ffn = attn, ffn
        self.attn_hc, self.ffn_hc = attn_hc, ffn_hc
        self.attn_norm, self.ffn_norm = attn_norm, ffn_norm
        self.engram = engram
        if type(checkpoint_attention) is not bool:
            raise ValueError("checkpoint_attention must be boolean")
        self.checkpoint_attention = checkpoint_attention
        self.freeze(keys=["attn_norm", "ffn_norm"], recurse=False)

    def __call__(self, h, pre, kv, engram_embedding):
        if self.engram is not None:
            h = self.engram(h, engram_embedding)
        attn_pre, post, comb = self.attn_hc(h)
        attn_input = rms_norm(hc_pre(h, pre), self.attn_norm)
        # Bound overlap with the MoE trace. The complete shared-state and input
        # VJP is required here: earlier adapters may affect both paths. Only
        # the separately verified outer frozen-prefix boundary prunes inputs.
        value, kv = (
            recompute(self.attn, attn_input, kv)
            if self.checkpoint_attention and self.training
            else self.attn(attn_input, kv)
        )
        h = hc_post(value, h, post, comb)
        pre, post, comb = self.ffn_hc(h)
        value = self.ffn(rms_norm(hc_pre(h, attn_pre), self.ffn_norm))
        return hc_post(value, h, post, comb), pre, kv


class DeepSeekTextModel(nn.Module):
    def __init__(
        self,
        path,
        *,
        rank=8,
        alpha=16.0,
        cache_mib=256,
        decoded_cache_mib=0,
        prefix_cache_mib=0,
        prefix_disk_cache=None,
        prefix_disk_cache_gib=0,
        activation_mode="checkpoint",
        chunk_size=512,
        checkpoint_blocks=True,
        adapter_layers=None,
        expert_adapter_layers=(),
        progress=None,
    ):
        super().__init__()
        self.activation_mode = validate_activation_mode(activation_mode)
        if any(
            type(v) is not int or v < 0 for v in (cache_mib, decoded_cache_mib, prefix_cache_mib)
        ):
            raise ValueError(
                "Packed, decoded and prefix cache limits must be nonnegative integer MiB"
            )
        root = Path(path).expanduser().resolve()
        disk_quota = validate_disk_options(prefix_disk_cache, prefix_disk_cache_gib)
        disk_files_before = {}
        if prefix_disk_cache is not None:
            from .sft import implementation_identity

            implementation = implementation_identity()
            destination = Path(prefix_disk_cache).expanduser().resolve()
            if destination == root or destination.is_relative_to(root):
                raise ValueError("Prefix disk cache must be outside the frozen model directory")
            disk_files_before = {
                p.resolve(): fingerprint(p)
                for p in root.rglob("*")
                if p.is_file() and p.suffix in {".safetensors", ".bin", ".u32", ".json", ".jinja"}
            }
        raw = json.loads((root / "config.json").read_text())
        c = raw["text_config"]
        if (
            raw["model_type"] != "deepseek_v41"
            or c["scoring_func"] != "sqrtsoftplus"
            or c["topk_method"] != "noaux_tc"
            or c["num_key_value_heads"] != 1
            or c["n_shared_experts"] != 1
            or c["rms_norm_eps"] != 1e-20
            or c["hc_mult"] != 4
            or c["index_topk"] != 512
            or c["kv_source_layer_ids"] != [2, 8, 14, 20]
            or c["num_hidden_layers"] != 40
        ):
            raise ValueError("Unsupported DeepSeek architecture configuration")
        if {k: raw["quantization"][k] for k in ("mode", "group_size", "bits")} != {
            "mode": "mxfp8",
            "group_size": 32,
            "bits": 8,
        }:
            raise ValueError("Expected MXFP8 streaming trunk")
        self.config = c
        self.checkpoint_blocks = checkpoint_blocks
        self._root = root
        self._archive = archive = TensorArchive(root)
        bank = ExpertBank(root)
        if bank.dims != {
            "hidden": c["hidden_size"],
            "inter": c["moe_intermediate_size"],
            "n_experts": c["n_routed_experts"],
            "n_layers": c["num_hidden_layers"],
        }:
            raise ValueError("Expert bank dimensions disagree with config")
        self._store = MatrixStore(
            bank,
            cache_bytes=cache_mib << 20,
            decoded_cache_bytes=decoded_cache_mib << 20,
            chunk_size=chunk_size,
        )
        self._engrams = EngramBank(root, c)
        ea = TensorArchive(root / "engram/engram-residents.safetensors")
        selected = set(range(c["num_hidden_layers"]) if adapter_layers is None else adapter_layers)
        expert_selected = set(expert_adapter_layers)
        if (
            not selected.issubset(range(40))
            or not expert_selected.issubset(range(40))
            or not (selected or expert_selected)
        ):
            raise ValueError("Select adapter layers in 0..39")
        self.adapter_config = {
            "rank": rank,
            "alpha": alpha,
            "adapter_layers": sorted(selected),
            "expert_adapter_layers": sorted(expert_selected),
            "activation_mode": self.activation_mode,
        }
        self.blocks = []
        for i in range(c["num_hidden_layers"]):
            prefix = f"layers.{i}"

            def native(name, *, activation_mode=None, **kwargs):
                return NativeLinear(
                    archive,
                    prefix + "." + name,
                    activation_mode=self.activation_mode
                    if activation_mode is None
                    else activation_mode,
                    **kwargs,
                )

            def tensor(name):
                return archive.array(prefix + "." + name)

            linears = {k: native("attn." + k) for k in ("wq_a", "wq_b", "wkv", "wo_b")}
            if i in selected:
                for name in ("wq_a", "wo_b"):
                    linears[name] = LoRALinear(linears[name], rank=rank, alpha=alpha)
            norms = {"q": tensor("attn.q_norm.weight"), "kv": tensor("attn.kv_norm.weight")}
            compressor = None
            if i in c["kv_source_layer_ids"]:
                compressor = {"wkv": native("attn.compressor.wkv")}
                if c["compress_ratios"][i] > 1:
                    compressor["wgate"] = native("attn.compressor.wgate")
                norms["compressor"] = tensor("attn.compressor.norm.weight")
            orank = c["o_lora_rank"]
            groups = [
                native("attn.wo_a", rows=(g * orank, (g + 1) * orank), quantize_input=False)
                for g in range(c["o_groups"])
            ]
            attn = ShortAttention(
                i,
                c,
                linears,
                norms,
                tensor("attn.attn_sink"),
                groups,
                compressor,
                indexer=(
                    FrozenIndexer(
                        i,
                        c,
                        {
                            name: native("attn.indexer." + name, activation_mode="checkpoint")
                            for name in (
                                ("wq_b", "weights_proj", "wk")
                                if i in c["kv_source_layer_ids"]
                                else ("wq_b", "weights_proj")
                            )
                        },
                        tensor("attn.indexer.k_norm.weight")
                        if i in c["kv_source_layer_ids"]
                        else None,
                    )
                    if i in c["index_source_layer_ids"]
                    else None
                ),
                activation_mode=self.activation_mode,
            )
            gate = DeepSeekGate(
                tensor("ffn.gate.weight"),
                tensor("ffn.gate.bias"),
                top_k=c["num_experts_per_tok"],
                route_scale=c["routed_scaling_factor"],
                norm_topk_prob=c["norm_topk_prob"],
            )
            shared = DeepSeekExpert(
                *(native(f"ffn.shared_experts.w{j}") for j in (1, 2, 3)),
                swiglu_limit=c["swiglu_limit"],
            )
            experts = []
            for e in range(c["n_routed_experts"]):
                projections = [
                    self._store.linear(f"{prefix}.ffn.experts.{e}.w{j}") for j in (1, 2, 3)
                ]
                if i in expert_selected:
                    projections = [LoRALinear(p, rank=rank, alpha=alpha) for p in projections]
                experts.append(DeepSeekExpert(*projections, swiglu_limit=c["swiglu_limit"]))
            ffn = DeepSeekMoE(gate, experts, shared)
            connections = [
                HyperConnection(
                    tensor(f"hc_{part}_fn"),
                    tensor(f"hc_{part}_scale"),
                    tensor(f"hc_{part}_base"),
                    hc_mult=c["hc_mult"],
                    sinkhorn_iters=c["hc_sinkhorn_iters"],
                    eps=c["hc_eps"],
                )
                for part in ("attn", "ffn")
            ]
            engram = None
            if i in c["engram_layer_ids"]:
                key = prefix + ".engram"
                engram = Engram(
                    NativeLinear(ea, key + ".wkv", activation_mode=self.activation_mode),
                    ea.array(key + ".q_weight"),
                    ea.array(key + ".k_weight"),
                )
            block = TrainingBlock(
                attn,
                ffn,
                *connections,
                tensor("attn_norm.weight"),
                tensor("ffn_norm.weight"),
                engram,
                checkpoint_attention=checkpoint_blocks,
            )
            mx.eval(block.parameters())
            self.blocks.append(block)
            if progress:
                progress(f"Loaded backbone layer {i + 1}/40")
        self.head = NativeLinear(archive, "head", activation_mode=self.activation_mode)
        self.norm = archive.array("norm.weight")
        self.freeze(keys=["norm"], recurse=False)
        self._progress = progress
        mx.eval(self.parameters())
        keys = [k for k, _ in tree_flatten(self.trainable_parameters())]
        if not keys or any(not k.endswith(("lora_a", "lora_b")) for k in keys):
            raise AssertionError("Only adapter parameters may enter the optimizer")
        # Private non-Module storage keeps cached activations out of parameter,
        # optimizer, and adapter/checkpoint traversal. No entry can cross the
        # earliest attention OR expert adapter, including a zero-initialized B.
        boundary = min(selected | expert_selected)
        disk = None
        self._prefix_disk_sources = {}
        if prefix_disk_cache is not None:
            if not boundary:
                raise ValueError("Prefix disk cache requires adapters strictly after layer 0")
            if progress:
                progress("Hashing complete frozen model payloads for exact prefix disk identity")
            payloads, observed = hash_payloads(root, disk_files_before, progress=progress)
            current_files = {
                p.resolve()
                for p in root.rglob("*")
                if p.is_file() and p.suffix in {".safetensors", ".bin", ".u32", ".json", ".jinja"}
            }
            if current_files != set(disk_files_before):
                raise RuntimeError("Frozen model file membership changed during disk-cache setup")
            if observed != disk_files_before or implementation_identity() != implementation:
                raise RuntimeError("Frozen model or implementation changed during disk-cache setup")
            self._prefix_disk_sources = observed
            self._prefix_disk_implementation = implementation
            binding = {
                "implementation": implementation,
                "platform": {
                    "system": platform.system(),
                    "machine": platform.machine(),
                    "release": platform.release(),
                    "version": platform.version(),
                    "device": mx.device_info()["device_name"],
                },
                "model_payloads": payloads,
                "activation_mode": self.activation_mode,
                "chunk_size": chunk_size,
                "frozen_blocks": boundary,
                "state_layout": {
                    k: c[k] for k in ("hidden_size", "hc_mult", "head_dim", "index_head_dim")
                },
                "prefix_config": c,
            }
            disk = FrozenPrefixDisk(prefix_disk_cache, quota_bytes=disk_quota, binding=binding)
        self._prefix_cache = FrozenPrefixCache(
            prefix_cache_mib << 20, frozen_blocks=boundary, disk=disk
        )
        self._prefix_signature = None
        self._prefix_sources = {}
        if self._prefix_cache.enabled:
            self._prefix_signature = self._frozen_prefix_signature()
            self._prefix_sources = {
                **archive.fingerprints,
                **ea.fingerprints,
                **self._store._fingerprints,
                **{file: fingerprint for file, _, fingerprint in self._engrams.layers.values()},
            }

    def __call__(self, tokens):
        return self.project_logits(self.hidden_states(tokens))

    def _cache_progress(self, message):
        if self._progress:
            stats = self._store.stats
            self._progress(
                f"{message}; cumulative packed reads/hits {stats.loads}/{stats.hits}; "
                f"decoded loads/hits/evictions {stats.decoded_loads}/{stats.decoded_hits}/{stats.decoded_evictions}; "
                f"retained packed/decoded {stats.cached_bytes / 2**30:.2f}/{stats.decoded_cached_bytes / 2**30:.2f} GiB"
            )

    def _frozen_prefix_signature(self):
        boundary = self._prefix_cache.frozen_blocks
        prefix = self.blocks[:boundary]
        if any(tree_flatten(block.trainable_parameters()) for block in prefix):
            raise RuntimeError("Prefix cache requires every prefix block to remain frozen")
        return (
            frozen_structure_signature(self.config),
            frozen_structure_signature(prefix),
            self._store.chunk_size,
        )

    def _validate_frozen_prefix(self):
        try:
            if getattr(self, "_prefix_disk_sources", {}):
                from .sft import implementation_identity

                if implementation_identity() != self._prefix_disk_implementation:
                    raise RuntimeError(
                        "Frozen prefix implementation changed; reconstruct the model"
                    )
            if self._frozen_prefix_signature() != self._prefix_signature:
                raise RuntimeError(
                    "Frozen prefix structure or weights changed; reconstruct the model"
                )
            for file, expected in getattr(self, "_prefix_disk_sources", {}).items():
                if fingerprint(file) != expected:
                    raise RuntimeError(f"Frozen prefix disk source changed: {file}")
            for file, expected in self._prefix_sources.items():
                stat = file.stat()
                if (stat.st_ino, stat.st_size, stat.st_mtime_ns) != expected:
                    raise RuntimeError(f"Frozen prefix checkpoint changed: {file}")
        except (OSError, RuntimeError):
            self._prefix_cache.clear()
            raise

    def clear_prefix_cache(self, *, reset_stats=False):
        """Release saved states; this does not authorize changing the frozen base.

        A cache belongs to this constructed model only. Never mutate frozen
        tensors in place: doing so cannot be detected by object/stat guards.
        Reconstruct the model after changing any base weight, prefix behavior,
        architecture, or adapter placement. Ordinary suffix adapter updates are
        safe and preserve reusable entries. This clears only RAM; opt-in disk
        entries remain immutable and bound to their construction identity.
        """
        self._prefix_cache.clear(reset_stats=reset_stats)

    def hidden_states(self, tokens):
        """Run the complete backbone and final norm, retaining the full context."""
        ids = np.array(tokens)
        c = self.config
        limit = min(MAX_CONTEXT_TOKENS, c.get("max_position_embeddings", MAX_CONTEXT_TOKENS))
        if ids.ndim != 2 or not 1 <= ids.shape[1] <= limit:
            raise ValueError(f"Expected 1..{limit} text tokens per sequence")
        hashes = self._engrams.hashes(ids)
        cache, key, saved, start = self._prefix_cache, None, None, 0
        # Inspect live trainability rather than relying on the construction-time
        # cache boundary. A manually unfrozen earlier block must never lose its
        # gradient, including when caching is disabled. Tokens/embedding lookups
        # and frozen Engram rows have no trainable input dependencies.
        first_trainable = None
        if self.checkpoint_blocks and self.training:
            # This implementation's embedding/Engram lookup banks are frozen
            # private stores; Engram projections belong to their block. Decline
            # the optimization if any other public parameter is made trainable,
            # even a downstream head/norm that would ordinarily be safe.
            trainable = tree_flatten(self.trainable_parameters())
            if trainable and all(name.startswith("blocks.") for name, _ in trainable):
                first_trainable = next(
                    (
                        i
                        for i, block in enumerate(self.blocks)
                        if tree_flatten(block.trainable_parameters())
                    ),
                    None,
                )
        if cache.enabled:
            self._validate_frozen_prefix()
            key = cache.key(ids, training=self.training, checkpoint_blocks=self.checkpoint_blocks)
            saved = cache.get(key)
        if saved is None:
            # Read only the embedding rows actually needed by this microbatch.
            unique, inverse = np.unique(ids, return_inverse=True)
            embedding = mx.concatenate(
                [self._archive.array("embed.weight", rows=(int(t), int(t) + 1)) for t in unique]
            )
            embedding = embedding[mx.array(inverse.astype(np.int32))].reshape(
                *ids.shape, c["hidden_size"]
            )
            h, pre = expand_streams(embedding, c["hc_mult"])
            kv = empty_attention_state(
                ids.shape[0], ids.shape[1], c["head_dim"], c["index_head_dim"], h.dtype
            )
        else:
            h, pre, kv = saved
            start = cache.frozen_blocks
            self._cache_progress(f"Exact frozen-prefix cache hit; skipped {start} backbone layers")
        for i in range(start, len(self.blocks)):
            block = self.blocks[i]
            bank_rows = mx.zeros((0,), dtype=h.dtype)
            if i in c["engram_layer_ids"]:
                bank_rows = self._engrams.gather(i, hashes[:, :, c["engram_layer_ids"].index(i)])
                mx.eval(bank_rows)
            if self.checkpoint_blocks and self.training:
                callback = (
                    (
                        lambda message, layer=i: self._cache_progress(
                            f"Layer {layer + 1}/40: {message}"
                        )
                    )
                    if self._progress
                    else None
                )
                h, pre, kv = recompute(
                    block,
                    h,
                    pre,
                    kv,
                    bank_rows,
                    progress=callback,
                    frozen_inputs=i > 0 and i == first_trainable,
                )
            else:
                h, pre, kv = block(h, pre, kv, bank_rows)
            mx.eval(h, pre, kv)
            if key is not None and saved is None and i + 1 == cache.frozen_blocks:
                self._validate_frozen_prefix()
                cached = cache.put(key, (h, pre, kv))
                if cached is not None:
                    h, pre, kv = cached
            if self._progress:
                self._cache_progress(
                    f"Forward layer {i + 1}/40; active MLX {mx.get_active_memory() / 2**30:.2f} GiB"
                )
        return rms_norm(hc_pre(h, pre), self.norm)

    def project_logits(self, hidden):
        """Project any contiguous or selected normalized hidden rows to logits."""
        return self.head(hidden).astype(mx.float32)

    def report(self):
        return {
            "checkpoint": str(self._root),
            "expert_source": self._store.checkpoint.source,
            "expert_manifest_sha256": self._store.checkpoint.manifest_sha256,
            "streaming": self._store.report(),
            "prefix_cache": self._prefix_cache.report(),
            "engram_bytes_read": self._engrams.bytes_read,
            "trainable_parameters": sum(
                v.size for _, v in tree_flatten(self.trainable_parameters())
            ),
            "peak_mlx_gib": mx.get_peak_memory() / 2**30,
            "activation_mode": self.activation_mode,
            "activation_gradient": activation_description(self.activation_mode),
            "context_limit": min(
                MAX_CONTEXT_TOKENS, self.config.get("max_position_embeddings", MAX_CONTEXT_TOKENS)
            ),
            "attention": "query-tiled sparse attention with frozen causal indexer and first-order VJP",
            "block_recomputation": self.checkpoint_blocks,
            "nested_attention_recomputation": self.checkpoint_blocks and self.training,
        }
