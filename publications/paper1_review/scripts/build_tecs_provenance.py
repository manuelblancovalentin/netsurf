#!/usr/bin/env python3
"""Bind the TECS-revision inputs, scripts, derived tables and figures by SHA-256 (offline)."""
import hashlib, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import tecs_grid_audit as ga  # noqa: E402

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

W = Path("/projects/netsurf/manuelbv/workspace/implementation-worktrees")
extra = {
    "kws_cross_pool_payload": W / "kws-transition-validation/workspace/runs/kws_transition_independent_pools_p05_derived_recovery_v1/scientific_payload.json",
    "distilbert_quantization_manifest": Path("/data/models/local--kona-assets--distilbert-sst2-int8/snapshots/1f07e0012b060f3658fa139839dab054478c5121/quantization_manifest.json"),
    "distilbert_moments_manifest": W / "transformer-integration/workspace/runs/distilbert_sst2_w8a8_reliability_v2_run1/selection/moments/manifest.json",
    "asset_readiness_manifest": Path("/projects/kona-assets/manuelbv/NETSURF_ASSET_READINESS.json"),
    "w8a8_rank_source": W / "w8a8-ber-stress/netsurf/w8a8_rank.py",
    "tflite_adapter_source": W / "m12-fkeras-multibench/netsurf/adapters/mlperf_tiny.py",
}
for m in ("resnet8", "kws", "toycar"):
    extra[f"{m}_mask_composition"] = ga.STUDIES[m]["path"].parent / "mask_composition.csv"
doc = {
    "schema": "paper1-tecs-revision-provenance/1.0",
    "experiments_executed": 0,
    "note": "Offline re-analysis of finalized studies. Source bundles were read, never written.",
    "source_trials": {m: {"path": str(s["path"]), "sha256": sha(s["path"])} for m, s in ga.STUDIES.items()},
    "other_sources": {k: {"path": str(p), "sha256": sha(p)} for k, p in extra.items()},
    "scripts": {p.name: sha(p) for p in sorted((ROOT / "scripts").glob("*tecs*.py"))},
    "derived_data": {p.name: sha(p) for p in sorted((ROOT / "data/tecs_audit").iterdir())},
    "figures": {p.name: sha(p) for p in sorted((ROOT / "figures").glob("tecs_*"))},
    "analytic_assets": {"source/figures/workflow.tex": sha(ROOT / "source/figures/workflow.tex")},
    "audit_parameters": {"thresholds": ga.THRESH, "signflip_draws": ga.SIGNFLIP_DRAWS, "seed": ga.SEED},
    "limits": [
        "One checkpoint, one evaluation cohort and one constructed mask per deterministic ranking and budget.",
        "Usefulness criterion fixed before regret was computed but after the data existed.",
        "Main-grid language-model cap frequency was never recorded.",
        "Residual-drift check covers DistilBERT only.",
    ],
}
out = ROOT / "data/tecs_revision_provenance.json"
out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
print("wrote", out)
