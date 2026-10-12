"""Experimental text-only, short-context DeepSeek V4.1 Flash trainer model.

Loads the mtplx streaming repack: MXFP8/BF16 trunk, real EXL3 routed experts,
and sparse MXFP8 Engram banks. All 40 backbone layers participate in text loss.
No vision, MTP, decode cache, long-context indexer, or PEFT export is claimed.
"""

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from .attention import ShortAttention
from .engram import Engram, EngramBank
from .hyperconnections import HyperConnection, expand_streams, hc_post, hc_pre, rms_norm
from .moe import DeepSeekExpert, DeepSeekGate, DeepSeekMoE
from .native import LoRALinear, NativeLinear
from .recompute import recompute
from .storage import ExpertBank, TensorArchive
from .streaming import MatrixStore


class TrainingBlock(nn.Module):
    def __init__(self, attn, ffn, attn_hc, ffn_hc, attn_norm, ffn_norm, engram=None):
        super().__init__()
        self.attn, self.ffn = attn, ffn
        self.attn_hc, self.ffn_hc = attn_hc, ffn_hc
        self.attn_norm, self.ffn_norm = attn_norm, ffn_norm
        self.engram = engram
        self.freeze(keys=["attn_norm", "ffn_norm"], recurse=False)

    def __call__(self, h, pre, kv, engram_embedding):
        if self.engram is not None:
            h = self.engram(h, engram_embedding)
        attn_pre, post, comb = self.attn_hc(h)
        value, kv = self.attn(rms_norm(hc_pre(h, pre), self.attn_norm), kv)
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
        chunk_size=512,
        checkpoint_blocks=True,
        adapter_layers=None,
        expert_adapter_layers=(),
        progress=None,
    ):
        super().__init__()
        root = Path(path).expanduser().resolve()
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
        self._store = MatrixStore(bank, cache_bytes=cache_mib << 20, chunk_size=chunk_size)
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
        }
        self.blocks = []
        for i in range(c["num_hidden_layers"]):
            prefix = f"layers.{i}"

            def native(name, **kwargs):
                return NativeLinear(archive, prefix + "." + name, **kwargs)

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
                i, c, linears, norms, tensor("attn.attn_sink"), groups, compressor
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
                    NativeLinear(ea, key + ".wkv"),
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
            )
            mx.eval(block.parameters())
            self.blocks.append(block)
            if progress:
                progress(f"Loaded backbone layer {i + 1}/40")
        self.head = NativeLinear(archive, "head")
        self.norm = archive.array("norm.weight")
        self.freeze(keys=["norm"], recurse=False)
        self._progress = progress
        mx.eval(self.parameters())
        keys = [k for k, _ in tree_flatten(self.trainable_parameters())]
        if not keys or any(not k.endswith(("lora_a", "lora_b")) for k in keys):
            raise AssertionError("Only adapter parameters may enter the optimizer")

    def __call__(self, tokens):
        ids = np.array(tokens)
        c = self.config
        if ids.ndim != 2 or not 1 <= ids.shape[1] <= c["index_topk"]:
            raise ValueError("Expected 1..512 text tokens per sequence")
        hashes = self._engrams.hashes(ids)
        # Read only the embedding rows actually needed by this microbatch.
        unique, inverse = np.unique(ids, return_inverse=True)
        embedding = mx.concatenate(
            [self._archive.array("embed.weight", rows=(int(t), int(t) + 1)) for t in unique]
        )
        embedding = embedding[mx.array(inverse.astype(np.int32))].reshape(
            *ids.shape, c["hidden_size"]
        )
        h, pre = expand_streams(embedding, c["hc_mult"])
        kv = mx.zeros((ids.shape[0], 0, c["head_dim"]), dtype=h.dtype)
        for i, block in enumerate(self.blocks):
            bank_rows = mx.zeros((0,), dtype=h.dtype)
            if i in c["engram_layer_ids"]:
                bank_rows = self._engrams.gather(i, hashes[:, :, c["engram_layer_ids"].index(i)])
                mx.eval(bank_rows)
            if self.checkpoint_blocks and self.training:
                h, pre, kv = recompute(block, h, pre, kv, bank_rows)
            else:
                h, pre, kv = block(h, pre, kv, bank_rows)
            mx.eval(h, pre, kv)
            if self._progress:
                self._progress(
                    f"Forward layer {i + 1}/40; active MLX {mx.get_active_memory() / 2**30:.2f} GiB; expert reads {self._store.stats.loads}"
                )
        return self.head(rms_norm(hc_pre(h, pre), self.norm)).astype(mx.float32)

    def report(self):
        return {
            "checkpoint": str(self._root),
            "expert_source": self._store.checkpoint.source,
            "expert_manifest_sha256": self._store.checkpoint.manifest_sha256,
            "streaming": self._store.report(),
            "engram_bytes_read": self._engrams.bytes_read,
            "trainable_parameters": sum(
                v.size for _, v in tree_flatten(self.trainable_parameters())
            ),
            "peak_mlx_gib": mx.get_peak_memory() / 2**30,
            "activation_gradient": "identity straight-through for FP8/FP4 activation quantization",
            "context_limit": 512,
            "block_recomputation": self.checkpoint_blocks,
        }
