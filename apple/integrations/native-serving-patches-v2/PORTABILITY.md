# Prerequisites and portability

The v2 checker verifies only the nine v2 target files. It does not establish the v1 plugin/settings changes, the host report correction, whole-tree identity, dependency compatibility, or hardware qualification.

Before applying v2, use the preserved sibling v1 checker to verify both disposable source roots in their v1 **after** state. From the parent integration directory:

```sh
python3 native-serving-patches-v1/check_sources.py --lane mlx-stream --root /absolute/disposable-stream --state after
python3 native-serving-patches-v1/check_sources.py --lane mlx-serve-host --root /absolute/disposable-host --state after
python3 native-serving-patches-v2/check_sources.py --root /absolute/disposable-stream --state before

git -C /absolute/disposable-stream -c core.hooksPath=/dev/null apply --check /absolute/integrations/native-serving-patches-v2/mlx-stream-portable.patch
git -C /absolute/disposable-stream -c core.hooksPath=/dev/null apply /absolute/integrations/native-serving-patches-v2/mlx-stream-portable.patch
python3 native-serving-patches-v2/check_sources.py --root /absolute/disposable-stream --state after
```

Run the v1 checks before the v2 application: some v2 files deliberately supersede their v1 hashes. A failed check requires an explicit port review, not forced application. Both checkers validate their targeted files only; together they still do not validate the entire source/dependency tree.

`check_sources.py` can run from the copied bundle using Python's standard library and supplied canonical source roots. The source paths recorded in `manifest.json` identify local evidence; the checker does not read those provenance paths. `test_patch_bundle.py` does read them to reconstruct a temporary tree and compare the complete patched files, so those five tests are local-evidence-dependent. Copying the bundle alone does not make that replay suite portable. No optional model inputs, reference helpers, binaries, or automatic build/launch machinery are included.

This note supplements the unchanged original README and author seal. Source and CPU/static evidence are distinct from full serving, adapter, compressed-path, long-context, and performance qualification.
