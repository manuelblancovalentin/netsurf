# Deprecated and historical material

Nothing in this directory is part of the canonical `netsurf` package. These
files are retained for provenance, compatibility, or recovery only.

- `legacy_source/netsurf/`: recovered 2025 implementation, loaded only by an
  explicit compatibility adapter.
- `historical_tutorials/`: original notebooks and saved outputs.
- `historical_docs/`: original README and manuscript copy.
- `historical_assets/`: images that were tracked by the historical repository.
- `historical_artifacts/root/`: formerly untracked root plots and the Nodus
  SQLite database.
- `env/`: snapshots of the historical Python environment.
- `external_links/`: ignored convenience links to external legacy datasets,
  checkpoints, results, synthesis material, and QKeras source.
- `recovery/`: ignored metadata from the empty Git initialization that replaced
  the deleted repository metadata before recovery.

Do not treat presence here as validation. Do not edit historical artifacts in
place. Maintained work belongs in the canonical source directories; disposable
generated work belongs in `workspace/`; finalized runs belong under the
configured external artifact root.
