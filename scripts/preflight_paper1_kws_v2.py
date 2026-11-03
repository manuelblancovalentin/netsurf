"""Validate the published-form-only KWS plan without executing fault trials.

The v1 campaign driver and its source fingerprint are intentionally untouched.
The researcher may run the validated v2 TOML with the pinned NetSuRF CLI later.
"""

import fcntl
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from workflows.paper1_campaign import (
    PINNED, ROOT, code_digest, read, sha, verified_stage, verify_run, write_new,
)
from netsurf.manifests import digest_object


def main() -> None:
    if Path(sys.executable).absolute() != Path(PINNED):
        raise ValueError(f"required interpreter: {PINNED}")
    plan = read(ROOT / "config/paper1_campaigns.json")["kws_v2"]
    config_path = ROOT / plan["configuration"]
    if sha(config_path) != plan["configuration_sha256"]:
        raise ValueError("KWS v2 TOML differs from the frozen plan")
    from netsurf.api import load_configuration
    from netsurf.adapters.registry import create_adapter
    from netsurf.samples import require_disjoint

    config = load_configuration(config_path)
    policy_names = [p.name for p in config.policies]
    estimator_names = [e.name for e in config.estimators]
    if len(policy_names) != len(set(policy_names)) or len(policy_names) != 7:
        raise ValueError("expected seven unique published-form KWS policies")
    if "fkeras_paper" not in policy_names or "fkeras_repository" in policy_names + estimator_names:
        raise ValueError("KWS v2 requires published-form FKeras only")
    n_ber = len(config.protection.probabilities)
    n_coverage = len(config.protection.coverages)
    n_fields = config.replications.count * config.execution.trials * n_ber
    n_eval = n_fields * (len(policy_names) * n_coverage + 2)
    n_pairs = n_coverage * (len(policy_names) + len(policy_names) * (len(policy_names) - 1) // 2)
    if (n_fields, n_eval, n_pairs) != (
        plan["fault_fields"], plan["mask_evaluations"], plan["comparison_family_per_ber_metric"]
    ):
        raise ValueError("recorded work or comparison family differs from the validated configuration")
    if n_eval * config.samples.evaluation_count != plan["inference_items"]:
        raise ValueError("recorded inference-item count differs from the configuration")
    if sha(Path(config.benchmark.checkpoint)) != plan["checkpoint_sha256"]:
        raise ValueError("checkpoint differs from frozen plan")
    if sha(Path(config.benchmark.asset_lock)) != plan["asset_lock_sha256"]:
        raise ValueError("asset lock differs from frozen plan")

    identity = {
        "schema_version": "paper1-kws-v2-preflight/1.0.0",
        "plan": plan,
        "source_sha256": code_digest(),
    }
    destination = ROOT / "workspace/paper1-campaigns" / config.name
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "execution.lock").open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        frozen = destination / "plan.json"
        if frozen.exists():
            if read(frozen) != identity:
                raise ValueError("KWS v2 campaign plan or source changed; do not resume")
        else:
            write_new(frozen, identity)

        def preflight(where: Path) -> None:
            baseline = verify_run(plan["exposure_source"])
            if baseline.scientific_payload_sha256 != plan["exposure_payload_sha256"]:
                raise ValueError("exposure-source payload differs from frozen plan")
            old = read(baseline.path / "scientific_payload.json")
            old = (old["replications"][0] if "replications" in old else old)["benchmark"]["partitions"]
            adapter = create_adapter(config, None)
            adapter.load()
            partitions = []
            for role, count in (
                ("ranking", config.samples.ranking_count),
                ("empirical_validation", config.samples.empirical_count),
                ("evaluation", config.samples.evaluation_count),
            ):
                _, _, partition = adapter.sample_partition(role, count)
                prior = old[role]
                if partition.source_split != prior["source_split"] or not set(partition.indices) <= set(prior["indices"]):
                    raise ValueError(f"{role} is not contained in an audited exposed source cohort")
                partitions.append(partition)
            require_disjoint(partitions)
            if len(adapter.bit_identities()) != plan["eligible_bits"]:
                raise ValueError("eligible bit population differs from frozen plan")
            write_new(where / "preflight.json", {
                "schema_version": "paper1-kws-v2-preflight/1.0.0",
                "partitions": [p.to_dict() for p in partitions],
                "exposure": "previously exposed fixed inputs; exploratory, not untouched confirmation",
                "representation": adapter.representation_manifest(),
                "ranking_policies": policy_names,
                "planned_fault_fields": n_fields,
                "planned_mask_evaluations": n_eval,
                "planned_inference_items": plan["inference_items"],
                "comparison_family": {
                    "pairing": "replication, logical_ber, trial, schedule_sha256",
                    "within_ber_per_metric_holm_tests": n_pairs,
                    "components_per_coverage": {
                        "policy_vs_unprotected": len(policy_names),
                        "policy_pairwise": len(policy_names) * (len(policy_names) - 1) // 2,
                    },
                },
                "plan_sha256": digest_object(identity),
            })

        verified_stage(destination / "preflight", preflight)
        print(json.dumps({
            "campaign": "kws_v2", "status": "preflight_verified",
            "config_sha256": plan["configuration_sha256"],
            "fault_fields": n_fields, "mask_evaluations": n_eval,
            "inference_items": plan["inference_items"],
            "within_ber_per_metric_holm_tests": n_pairs,
            "receipt": str(destination / "preflight/STAGE_COMPLETE.json"),
        }, indent=2))


if __name__ == "__main__":
    main()
