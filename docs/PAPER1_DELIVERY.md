# NetSuRF Paper 1 — scientific reconstruction delivery

2 November 2026. **First delivery gate only: no replacement manuscript yet.**

## Read in this order

1. [Original-to-current audit](PAPER1_ORIGINAL_TO_CURRENT_AUDIT.md): full original
   manuscript reading, all eight figure mappings, encoding/qPolarGrad/hardware
   discrepancies, central claim and five proposed figures.
2. [Evidence inventory](PAPER1_EVIDENCE_INVENTORY.md): which complete engineering
   curves already exist, exact source identities, computation costs and limitations.
3. [Novelty and hostile-review audit](PAPER1_NOVELTY_AUDIT.md): primary-method
   comparisons, including the newly supplied EISFINN and FCCM papers; baseline
   readiness and concrete fixes for submission risks.
4. [Next steps](PAPER1_NEXT_STEPS.md): engineering manuscript structure, optional
   additional-week experiments, explicit operation counts and decisions to freeze.

The defensible story is an engineering investigation of how well static encoded-bit
rankings guide budgeted protection under simultaneous parameter faults. The
published FCCM flow is acknowledged as prior work. Neither a new universal score
nor superiority of a conditional optimizer is established. Hardware overheads are
not measured by the current logical-suppression experiments.

## Included evidence and checks

The package includes six plot previews (PNG/PDF), all-condition and paired tables,
verification receipts, source inventory, and offline reproduction scripts/tests.
All are derived from existing immutable evidence; no model inference was executed.

- 125 finalized records, 592 consumed files and canonical scientific payloads
  rechecked against original inventories.
- 7,616 static trial rows, 280 condition summaries and 1,680 paired summaries
  checked across all configured replications.
- Exact schedule pairing, event accounting, clean/full controls, protected
  cardinalities and recorded split disjointness checked.
- Four targeted offline tests passed, including deliberately different replication
  outcomes, corruption detection and a complete analytical counterexample.
- Six plots inspected for readable labels, uncertainty, controls and task units.

The scope is the listed consumed evidence, not every historical file. Large mask
files were hash-checked; all individual mask identities were not re-enumerated in
this offline pass. Saved permutation/Holm values are integrity-checked source
values, not newly recalculated permutation tests. No full ML regression suite was
run: neither scientific execution code nor model behavior changed.

Source papers are cited and identified by checksum, but copyrighted full PDFs,
models, datasets, giant prediction tables and old manuscript projects are not
redistributed in this package. Prior run paths remain locators; reproducing the
audit requires access to those original artifacts on Kona.

## Offline commands

```bash
cd /projects/netsurf/manuelbv/workspace/implementation-worktrees/paper1-engineering
PYTHONDONTWRITEBYTECODE=1 /mnt/raid0/ml/conda_envs/manuelbv/envs/netsurf/bin/python -m pytest -q \
  tests/test_paper1_evidence_audit.py -o cache_dir=workspace/cache/pytest-paper1

/mnt/raid0/ml/conda_envs/manuelbv/envs/netsurf/bin/python workflows/paper1_evidence_audit.py \
  --sources /projects/netsurf/manuelbv/publications/tecs_internal_review_2026-11-02/evidence/sources.json \
  --output workspace/runs/paper1_audit_independent_check

/mnt/raid0/ml/conda_envs/manuelbv/envs/netsurf/bin/python workflows/paper1_curve_preview.py \
  workspace/runs/paper1_audit_independent_check
```

Use a new output directory each time. These commands require the already installed
NumPy, SciPy, Matplotlib and pytest; they do not install anything or import TensorFlow.
The exported `evidence/source_locators.json` is a portable copy of the locator
receipt used above, not a substitute for checking the original finalized bundles.

## Archive and Mac transfer

The delivered archive is:

`/projects/netsurf/manuelbv/workspace/implementation-worktrees/paper1-engineering/workspace/exports/NetSuRF_Paper1_Reconstruction_2026-11-02.zip`

It contains `PACKAGE_MANIFEST.json` with the source commit and SHA-256 for every
other entry. On a Mac with an existing SSH host alias `kona`:

```bash
scp kona:/projects/netsurf/manuelbv/workspace/implementation-worktrees/paper1-engineering/workspace/exports/NetSuRF_Paper1_Reconstruction_2026-11-02.zip ~/Downloads/
unzip -t ~/Downloads/NetSuRF_Paper1_Reconstruction_2026-11-02.zip
```

Replace `kona` with the researcher's existing SSH host/alias if different. This is
not a claim to have tested the Mac's SSH configuration.

Work is on isolated branch `paper1-engineering-reconstruction`; no merge or push.
Canonical `v2`, the previous manuscript workspace, asset locks, researcher TOMLs
and all historical scientific results remain unchanged by this task.
