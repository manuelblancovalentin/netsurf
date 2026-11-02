# Internal-review delivery validation

Date: 2 November 2026. Scope: manuscript and offline derived analysis only.

## Checks completed

- Reverified 125 finalized source records and SHA-256 for 592 consumed files,
  including canonical scientific payload digests. Scope is the consumed
  evidence, not every unused historical file.
- Recomputed 7,616 static raw rows into 280 condition summaries and checked
  1,680 paired summary rows using the full schedule identity. Verified event
  accounting, zero/full controls and recorded partition disjointness. Existing
  large masks were checksum-verified, not recreated.
- Built all figures from those saved data; no checkpoint loading, training,
  gradient computation, mask construction or fault inference was performed.
- `latexmk -pdf main.tex`: successful **18-page** ACM manuscript, including
  six figures, four numbered tables, an analytical risk table and references.
- No undefined citations/references or overfull boxes in the final log. All 12
  bibliography entries are cited. Remaining underfull-spacing diagnostics are
  cosmetic; the ACM reference-format warning is intentional for an internal
  draft without fabricated publication metadata.
- Rendered the complete PDF and inspected the main figure/table pages. Fixed
  legend collisions and an intermediate vector-embedding label problem. Final
  figures use the matching 400-dpi raster output; vectors are also supplied.
- **10 bounded offline tests passed**, including raw replication/pairing checks,
  hash-corruption rejection, exact analytical counterexample, INT8 worked
  example, full-tier equivalence, bibliography coverage and manuscript labels.

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 \
 /mnt/raid0/ml/conda_envs/manuelbv/envs/netsurf/bin/python -m pytest -q \
 tests/test_paper1_evidence_audit.py tests/test_paper1_manuscript.py \
 -p no:cacheprovider
```

The scientific engine regression/integration suite was not rerun: no engine
source changed, and this manuscript task expressly avoids new scientific runs.
This is not a claim that all framework/backend tests were revalidated here.

## Preservation

Canonical `v2` remains at `e3209f7f67ef8f30cebb0eed791bf817164d39c0` with its
researcher-owned untracked files intact. Manuscript work is isolated on
`paper1-engineering-reconstruction`. No previous draft, finalized bundle, asset
lock, experiment configuration, running campaign, environment or toolchain was
modified. No merge or push was performed.

The export manifest records the delivery source commit and hashes of packaged
files. ZIP members are CRC-checked when packaged. Scientific payload hashes are
not replaced by presentation-file hashes.
