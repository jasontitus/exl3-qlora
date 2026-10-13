# Exact frozen-prefix cache review — 2026-10-12

Implementation is staged in `src/exl3_mlx/prefix_cache.py` and `deepseek.py`.
**21 focused tests pass: 11 CPU cases and 10 coordinated Metal cases, in 2.31
seconds.** Receipt: `reports/prefix-cache-final-tests.xml`. Ruff passes. The
process-review agent independently inspected the final implementation and
reported no additional blocker. The active experiment's immutable runtime was
not changed. The parent is running the complete suite before freezing runtime-v2.

## Scope and correctness contract

`DeepSeekTextModel(prefix_cache_mib=0)` preserves the existing uncached default.
The first adapted block is the minimum of attention and expert adapter block
IDs. Only blocks strictly before it may be cached; zero-initialized adapters do
not count as frozen. Selecting block 0, including the all-40 default, makes the
cache inactive even with a positive budget.

Each entry represents the entire exact token-ID batch and shape. Canonical
int64 token bytes are retained in the key, so different sequences cannot collide
through a digest or common prefix. Training/evaluation and block-recomputation
modes use separate keys. Current frozen prefix blocks have no stochastic dropout;
the mode separation is conservative. Cached state contains materialized `h`,
the next pre-mix, compressed KV, index keys, selected indices, and packed
candidate blocks. No route or activation from the trainable suffix is cached.
The suffix executes normally on every forward and backward.

The cache is a private, non-Module object belonging to one constructed model.
Its arrays do not enter model parameters, optimizer state, or adapter exports.
It is not serialized and does not persist across training-process restarts.
Ordinary suffix adapter updates or loads preserve entries. A resumed process
starts with a cold cache. `clear_prefix_cache(reset_stats=False)` releases entries
without changing the immutable-base contract.

## Bounds and invalidation

The LRU bound includes every retained tensor's `nbytes` and the full token-key
payload. It excludes Python object overhead, allocator/OS caches, source tensors,
temporary materialization copies, and entries still referenced by an executing
caller. Oversized states bypass admission without emptying useful entries or
performing a needless materialization. Admitted states evict before copying.
This byte limit is a component of the process memory budget, not an OS quota.

Before lookup and insertion, the model checks frozen-prefix trainability,
public module/configuration structure, retained array-storage/shape/strides/dtype identities,
decoder chunk size, and native/expert/Engram source-file stat fingerprints.
A detected change clears entries and fails. Reconstruction is required after a
frozen base or architecture change.

MLX creates temporary Python wrappers even for frozen arrays while tracing
`nn.value_and_grad`. The initial object-ID guard falsely rejected valid training.
The corrected guard uses a zero-copy NumPy view of evaluated storage (a uint16
view for BF16) and keeps the original view owner alive. Retention prevents an
allocator from recycling the original address for different weights. The guard
accepts identical storage aliases and rejects actual replacement. This adds no
weight copy or change to the forward/backward arithmetic.

**The caller must never modify a frozen tensor's storage in place, change private
prefix behavior, or mutate cached arrays.** Storage/stat guards do not hash all
weight bytes or detect every in-place/private mutation. This limitation is
explicit in the API documentation. The supported engine treats frozen arrays
and prefix blocks functionally; optimizer updates touch only suffix adapters.

## Verification and remaining evidence

Completed CPU tests cover exact keys and modes; all six stored tensor leaves;
byte accounting and independent copies; eviction before materialization;
oversized bypass; disabled and first-block-adapter inactivity; clear/reset;
replacement/configuration signatures; and invalid limits.

The completed GPU tests use the real `hidden_states`, recomputation, and Adam paths
with a small deterministic trunk. They demand bitwise FP32/BF16 loss, gradients,
adapter values, and optimizer-state equality on a cold miss and later hits across
three updates. Additional cases cover an earlier expert adapter boundary,
suffix adapter reload, changed token IDs, separate evaluation keys, trace-stable
frozen storage identities, and rejection
of frozen weight/trainability/configuration/file changes. Those fixtures are not
a substitute for full-checkpoint cache parity.

The previously captured final-block diagnostic uses the same mathematical
isolation, but this reusable implementation still needs its own actual-model
check. No end-to-end speedup or actual-checkpoint byte parity is claimed yet.
Repeated examples that fit in the LRU can reuse work. Unique batches, changed
token sequences during generation, separate processes, and data exceeding the
LRU can require complete prefix passes. Separate train/eval keys mean an
evaluation warm-up does not warm training's key.

## Related adversarial findings

The independent composed-gradient replay already passed 19 NumPy finite-
difference checks and direct-versus-recompute comparisons. That reduces generic
composition-defect risk, but does not establish why the original real all-layer
step worsened NLL. This cache changes computation reuse, not training arithmetic
or the claimed gradient. It must not be represented as a quality fix before real
fit/evaluation evidence exists.

The standalone frozen-prefix script now binds its own SHA into cache/resume
identity, separates base from resumed-initial scores, and rejects output under
the frozen model directory, addressing review feedback. Standard SFT selected-
layer configuration and checkpoint checks were coherent in read-only review.
The Studio trailing-comma parsing issue was sent to its owner for correction.
