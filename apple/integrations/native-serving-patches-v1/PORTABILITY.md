# Portability of this source-patch bundle

The patch files, manifest, checker and notices are copied unchanged from the independently reviewed local bundle. The manifest intentionally retains historical absolute source and evidence paths so their original hashes and provenance remain intact.

`check_sources.py` reads the manifest and relative patch beside itself, then verifies the five targeted files under the caller-supplied canonical absolute source roots. It does not read `source_before`, `source_after` or `evidence_sha256` paths and can check another location containing those exact modified target bytes. That does not certify the remainder of that checkout, its dependencies, build, or runtime.

The full replay tests do read the manifest's historical `source_before` and `source_after` files. They require this workspace's retained evidence or an explicitly reviewed relocation of it. This product directory is not a standalone source snapshot, a reconstruction of the modified bases from their upstream ancestor commits, or a self-contained replay/evidence archive. Do not rewrite those provenance paths silently or claim a stock upstream port.

No engine source, binary, dependency tree or model is installed by copying this bundle. Applying it, rebuilding and evaluating a resulting runtime are separate operations. Preserve all existing relevant notices when integrating.
