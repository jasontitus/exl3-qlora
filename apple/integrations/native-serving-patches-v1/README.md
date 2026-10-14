# Exact local engine integration patches

This bundle contains two patches affecting five production files, not another engine distribution. It reproduces the approved changes to this project's existing modified mlx-stream / mlx-serve host stack. It does not establish compatibility with a clean upstream checkout. Nothing here activates a runtime, installs dependencies, loads a model, or runs a GPU job.

`mlx-stream.patch` changes four DeepSeek files. Exact generation retains the device embedding through prefill/decode handover; loader/init flags must agree; ordinary execution keeps its prior embedding fence. The existing generation-state checks remain. The weights map owns the retained tensor until teardown. Decode billing includes retained embedding bytes before wiring calculations, and handover no longer credits those bytes as freed. The generic unbound `plugin.bill(req)` advisory path is not certified equivalent to the exact scheduler `loadBytes` / Module path.

`mlx-serve-host.patch` changes only `src/exact_generation.zig`. It emits the existing frontend's six execution-profile fields: deadline, environment, process memory, native box, physical memory and reserve. It replaces the obsolete `memory_cap_gib` field; changing only its lookup would still produce an incompatible result. Generation arithmetic, termination, routes and resource enforcement are unchanged.

The report-builder CPU tests exercised actual generic native `load`/`run` with a synthetic tiny generator and the real Python result validator, including missing/obsolete-field failures. The actual single absent-adapter request later completed with the repaired binary and released its own lease. This is a 33-prompt-token/four-generated-token regression, not adapter qualification, trainer-forward equivalence, a performance result, or a general memory-fit guarantee. `manifest.json` pins the source, CPU, build and actual-run evidence. The current 150 evaluation and training snapshots are unrelated and untouched.

## Bases and reproduction

Use separate disposable copies of the exact previously modified source roots. The stream patch's root is the prior profile `prototype/`, not its parent directory. The host patch's root contains `src/exact_generation.zig`. Full local source/capture/build manifests are hash-bound in `manifest.json`; per-file before/after SHA-256 values and canonical target-file fingerprints identify the five precise inputs. Ancestor Git IDs identify history only: these bases also contain prior local adapter, observer, baseline-null, cache/depth and lifecycle integrations. Do not treat the ancestor IDs as sufficient reconstruction instructions.

Run from this bundle directory, replacing the two example absolute paths with your disposable source roots:

```sh
python3 check_sources.py --lane mlx-stream --root /absolute/disposable-stream --state before
python3 check_sources.py --lane mlx-serve-host --root /absolute/disposable-host --state before

git -C /absolute/disposable-stream -c core.hooksPath=/dev/null apply --check /absolute/bundle/mlx-stream.patch
git -C /absolute/disposable-host -c core.hooksPath=/dev/null apply --check /absolute/bundle/mlx-serve-host.patch

git -C /absolute/disposable-stream -c core.hooksPath=/dev/null apply /absolute/bundle/mlx-stream.patch
git -C /absolute/disposable-host -c core.hooksPath=/dev/null apply /absolute/bundle/mlx-serve-host.patch

python3 check_sources.py --lane mlx-stream --root /absolute/disposable-stream --state after
python3 check_sources.py --lane mlx-serve-host --root /absolute/disposable-host --state after
```

A failed before check means the base differs. Do not force offsets, silently merge, or assume a partially applied patch is correct. Review an explicit port to any other base. The checker is read-only and checks targeted files, not the whole dependency tree or binary compatibility. The replay tests apply both full patches only in temporary directories and compare complete resulting files to the approved sources.

For an eventual local product commit, keep the two source-root changes distinct, preserve their existing notices, run the package's relevant CPU tests and the previously reviewed build process, then separately review any produced runtime and concrete request. No commit or public push was performed in producing this bundle. The bundle intentionally omits binaries, weights, the unrelated 993-file host tree, and automatic build/launch scripts.

## License provenance

`licenses/mlx-stream-LICENSE` and `licenses/mlx-serve-host-LICENSE` contain the MIT notices read from the exact ancestor Git objects recorded in `manifest.json`; the repositories and source heads came from the captured local source history. These notices are provenance, not a claim that the locally modified trees are byte-identical to upstream. Preserve upstream and any additional existing dependency notices when integrating. This bundle neither redistributes the dependency trees nor changes their licenses.
