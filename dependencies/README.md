# Dependencies

`pyproject.toml` is authoritative for the maintained core. Historical Conda and
pip snapshots are preserved under `DEPRECATED/env/`; they are provenance, not a
portable lock file. Optional framework adapters must document and validate their
own dependency boundary before being added to the core environment.
