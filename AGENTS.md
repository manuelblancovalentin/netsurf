# NetSuRF repository working rules

## Scientific authority

- The historical repository and `NETSURF_AUDIT_2026-10-21.md` are evidence, not an automatically valid specification.
- Never silently redefine qPolarGrad. Use `legacy_qpolargrad` for the recovered expression and a distinct name for any proposed replacement.
- Keep susceptibility estimators, protection policies, fault models, and mitigation guarantees separate.
- Logical BER is not a physical SEU rate. Detection is not correction. Synthesis is not fabricated silicon.
- Any publication number must be generated from a finalized, hashed result manifest.

## Preservation

- Do not edit historical checkpoints, datasets, rankings, CSV results, notebooks, HLS workspaces, synthesis reports, or `DEPRECATED/external_links/qkeras` in place.
- Do not reset, clean, or discard the dirty handoff state. Its tracked diff SHA-256 is recorded in `docs/LEGACY_MAPPING.md`.
- New results belong under `workspace/runs/` by default and are write-once after finalization. Historical external archives remain untouched.
- Do not commit proprietary libraries, credentials, license data, or private environment details.

## Development

- New scientific code lives in the side-effect-free canonical `netsurf` package. Importing it must not initialize legacy Nodus state.
- Add a deterministic test or documented validation plan for every scientific behavior change.
- Use the Python 3.11 `netsurf` environment through the private `proj` project `netsurf`.
- Use Wolf only for hardware stages it actually supports. Do not represent the ML experiment engine as a Wolf feature.
- Do not launch training, large Monte Carlo campaigns, HLS, synthesis, or physical design without a bounded command, prerequisites, output path, and resource estimate.

## Current bounded milestone

Run `python -m pytest -q` in the activated `netsurf` project. Use the bounded
schema-4.3 commands in `docs/REPRODUCIBILITY.md`; do not retrain checkpoints or
launch `mnist_conditional_screening_research` automatically. Preserve search,
natural-fault, conditional-background, protection-budget, and replication
uncertainty separately. Polar and factorized Kronecker interactions are
proposal diagnostics, not accepted protection theories; candidate marginals
are not exact swap effects; small-pool optima are not whole-network optima.

## Task log (kona standard)
`kona-task` is installed at `/usr/local/bin/kona-task`, which may not be on an agent's PATH.

This project keeps its tasks as files, one per task, in `.kona/tasks/` of your own copy of the project. You write them as you work, so nobody has to type them into the website later. The website reads them.

* Start of any piece of work that is more than a few minutes: `kona-task new --title "<one line>" --status doing --commit`. Use `--depends <id>` when it waits for another task, `--end YYYY-MM-DD` when it has a due date, `--tag`, `--stage`, `--target`, `--action`, `--lane`, `--milestone` with the words this project already uses (look at `kona-task list` and at older task files first; do not invent new words when one fits).
* When its state changes: `kona-task set <id> --status blocked|doing|done [--percent N] --commit`. Say what blocked it in `--notes`. Mark `done` only when the result exists on disk and you looked at it; write what you verified in `--notes`.
* Never edit another person's task files, never write outside `.kona/tasks/`, never put a secret, a password, a license server, a private path or another person's data in a task. The id is `<project>-<login>-<number>` and the command makes it: do not choose or rename ids by hand.
* Commit the task files together with the work they describe (`--commit` does it for the task file; never push unless the owner asks). Run `kona-task check` before you finish: it must print `0 problems`.
* If the command is missing, say so; do not make task files by hand.
