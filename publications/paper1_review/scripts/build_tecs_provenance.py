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
run_a = json.loads((ROOT / "data/tecs_runA/runA_analysis_provenance.json").read_text())
run_b = json.loads((ROOT / "data/tecs_runB/runB_tabulation_provenance.json").read_text())
executed = []
for model in ("resnet8", "kws"):
    r = run_a[model]
    executed.append({"run": f"Run A {model}", "run_dir": r["run_dir"], "run_id": r["run_id"], "source_commit": r["source"]["commit"],
                     "scientific_payload_sha256": r["scientific_payload_sha256"], "trials_sha256": r["trials_sha256"],
                     "config_sha256": r["config_sha256"], "checks": r["checks"], "event_identity_violations": r["event_identity_violations"]})
r = json.loads((ROOT / "data/tecs_runA2/runA2_analysis_provenance.json").read_text())["kws"]
executed.append({"run": "Run A2 kws", "run_dir": r["run_dir"], "run_id": r["run_id"], "source_commit": r["source"]["commit"],
                 "scientific_payload_sha256": r["scientific_payload_sha256"], "trials_sha256": r["trials_sha256"],
                 "config_sha256": r["config_sha256"], "checks": r["checks"], "event_identity_violations": r["event_identity_violations"]})
executed.append({"run": "Run L2 resnet8 correction mechanisms", "script": "workflows/correction_mechanisms_resnet8.py", "commit": "ea3cc98",
                 "outputs": "data/tecs_runL2 (64 write-once per-draw files)", "finalized_by_engine": False})
for variant in ("pruned_s05", "pruned_s07", "undertrained_e100") + tuple(f"retrained_seed{s}" for s in (7101, 7102, 7103, 7104, 7105)):
    r = json.loads((ROOT / "data/tecs_runK" / variant / "provenance.json").read_text())["resnet8"]
    executed.append({"run": f"Run K resnet8 {variant}", "run_dir": r["run_dir"], "run_id": r["run_id"], "source_commit": r["source"]["commit"],
                     "scientific_payload_sha256": r["scientific_payload_sha256"], "trials_sha256": r["trials_sha256"],
                     "config_sha256": r["config_sha256"], "checks": r["checks"], "event_identity_violations": r["event_identity_violations"]})
r = json.loads((ROOT / "data/tecs_runN/provenance.json").read_text())
executed.append({"run": r["run"], "script": r["script"], "commit": r["script_commit"], "outputs": "data/tecs_runN (analysis and 12 evaluation shards)",
                 "finalized_by_engine": False})
for model in ("pythia", "qwen"):
    r = run_b[model]
    executed.append({"run": f"Run B {model}", "replay_dir": r["replay_dir"], "replay_plan_sha256": r["replay_plan_sha256"],
                     "observation_listing_sha256": r["observation_listing_sha256"], "observations": r["observations"]})
REPO = ROOT.parents[1]
doc = {
    "schema": "paper1-tecs-revision-provenance/1.1",
    "experiments_executed": len(executed),
    "executed_runs": executed,
    "note": "Offline re-analysis of finalized studies, plus two approved runs of 11 November 2026: Run A (TFLite confirmation, "
            "two finalized engine bundles) and Run B (bitwise replay of the saved language-model fields with cap statistics). "
            "Source bundles were read, never written.",
    "derived_data_runA": {p.name: sha(p) for p in sorted((ROOT / "data/tecs_runA").iterdir())},
    "derived_data_runB": {p.name: sha(p) for p in sorted((ROOT / "data/tecs_runB").iterdir())},
    "derived_data_runA2": {p.name: sha(p) for p in sorted((ROOT / "data/tecs_runA2").iterdir())},
    "derived_data_runL2": {p.name: sha(p) for p in sorted((ROOT / "data/tecs_runL2").iterdir())},
    "derived_data_runK": {str(p.relative_to(ROOT / "data/tecs_runK")): sha(p) for p in sorted((ROOT / "data/tecs_runK").rglob("*")) if p.is_file()},
    "derived_data_runN": {str(p.relative_to(ROOT / "data/tecs_runN")): sha(p) for p in sorted((ROOT / "data/tecs_runN").rglob("*")) if p.is_file()},
    "workflow_scripts": {name: sha(REPO / "workflows" / name) for name in ("w8a8_cap_replay.py",)},
    "frozen_plans": {name: sha(ROOT / name) for name in ("review/RUN_A_B_FROZEN_PLANS_2026-11-11.md", "review/RUN_A2_D_F_FROZEN_PLANS_2026-11-11.md",
                                                       "review/RUN_J_K_L2_FROZEN_PLANS_2026-11-11.md", "review/RUN_N_METHOD_CAMPAIGNS_FROZEN_PLAN_2026-11-11.md")},
    "source_trials": {m: {"path": str(s["path"]), "sha256": sha(s["path"])} for m, s in ga.STUDIES.items()},
    "other_sources": {k: {"path": str(p), "sha256": sha(p)} for k, p in extra.items()},
    "scripts": {p.name: sha(p) for p in sorted((ROOT / "scripts").glob("*tecs*.py"))},
    "derived_data": {p.name: sha(p) for p in sorted((ROOT / "data/tecs_audit").iterdir())},
    "figures": {p.name: sha(p) for p in sorted((ROOT / "figures").glob("tecs_*"))},
    "analytic_assets": {"source/figures/workflow.tex": sha(ROOT / "source/figures/workflow.tex")},
    "audit_parameters": {"thresholds": ga.THRESH, "signflip_draws": ga.SIGNFLIP_DRAWS, "seed": ga.SEED},
    "limits": [
        "One trained checkpoint per model. Outside Run A: one evaluation cohort and one constructed mask per deterministic ranking and budget.",
        "Run A varies calibration draw, masks, evaluation window and fault fields jointly; DS-CNN has three new replications and does not meet the decision rule.",
        "Usefulness criterion fixed before regret was computed but after the main-grid data existed.",
        "Language-model cap frequency comes from a replay of the saved fields (Run B), not from the original runs.",
        "Residual-drift check covers DistilBERT only.",
    ],
}
out = ROOT / "data/tecs_revision_provenance.json"
out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
print("wrote", out)
