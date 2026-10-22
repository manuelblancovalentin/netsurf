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
- New results belong under the configured external artifact root and are write-once after finalization.
- Do not commit proprietary libraries, credentials, license data, or private environment details.

## Development

- New scientific code lives in the side-effect-free canonical `netsurf` package. Importing it must not initialize legacy Nodus state.
- Add a deterministic test or documented validation plan for every scientific behavior change.
- Use the Python 3.11 `netsurf` environment through the private `proj` project `netsurf`.
- Use Wolf only for hardware stages it actually supports. Do not represent the ML experiment engine as a Wolf feature.
- Do not launch training, large Monte Carlo campaigns, HLS, synthesis, or physical design without a bounded command, prerequisites, output path, and resource estimate.

## Current bounded milestone

Run `python -m tests.run` in the activated `netsurf` project, then follow `docs/REPRODUCIBILITY.md` for the twice-repeated MNIST-lite experiment.
