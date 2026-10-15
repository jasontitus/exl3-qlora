# M5 interior grid-sampling precision test

Eight dispatches completed on M5 Max with unchanged sources and tolerances. Independent audit reconstructed 1,024 rational reference values and replayed all 2,048 output values.

| Input/grid dtype | Before failures /256 | After failures /256 |
|---|---:|---:|
| half/half |170|0|
| half/float |44|0|
| float/half |248|0|
| float/float |4|4|

The float/float outputs are identical before and after. Indices 24,34,191,197 exceed the original tolerance in both; the cause remains unresolved. No tolerance was changed. This is partial correctness evidence for a single source lineage with a bounded interior fixture, not general grid-sampling qualification or a speed result. No borders, BF16 or generic-shape claim.

Peak observed owned RSS was 115,359,744 bytes; exact owned release and terminal process closure passed. See [audited evidence](evidence.json). Training is not activated. A draft may use only the explicitly qualified half-involving scope and must retain the FP32 limitation in provenance.
