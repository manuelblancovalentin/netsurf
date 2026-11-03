"""Read-only check of cross-precision manuscript receipts against finalized runs.

No model inference, source-bundle writes, mask generation or fault replay.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
RECEIPT = json.loads((HERE / "evidence/targeted_cross_precision_receipts.json").read_text())
SOURCE = Path(RECEIPT["source_root"])
FROZEN_PLANS = SOURCE.parents[1] / "config/experiments"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual: float, expected: float, tolerance: float = 1e-5) -> None:
    assert math.isclose(actual, expected, abs_tol=tolerance, rel_tol=0), (actual, expected)


def verify_common(record: dict) -> tuple[Path, dict]:
    run = SOURCE / record["run"]
    analysis_file = run / "analysis.json"
    assert digest(analysis_file) == record["analysis_file_sha256"]
    analysis = json.loads(analysis_file.read_text())
    assert analysis["plan_sha256"] == record["plan_sha256"]
    assert analysis["payload_sha256"] == record["scientific_payload_sha256"]
    source_zip = SOURCE / (record["run"] + "_export") / "complete.zip"
    assert digest(source_zip) == record["complete_export_zip_sha256"]
    return run, analysis


def verify_distilbert() -> None:
    record = RECEIPT["distilbert_independent_validation"]
    run, analysis = verify_common(record)
    plan = json.loads((FROZEN_PLANS / (record["run"].removesuffix("_run1") + ".json")).read_text())
    assert plan["eligible_bits"] == 1024
    assert plan["selection_source_indices"] == list(range(288, 544))
    assert plan["final_source_indices"] == list(range(544, 800))
    for budget, expected in record["paired_significance_minus_isolated_nats"].items():
        row = next(c for c in analysis["comparisons"]
                   if c["ber"] == 16 / 1024 and c["budget"] == int(budget)
                   and c["comparison"] == "exact_isolated_vs_significance")
        close(row["mean_baseline_minus_isolated"], expected["mean"], 1e-12)
        for actual, target in zip(row["ci95_percentile_bootstrap"], expected["ci95"]):
            close(actual, target, 1e-12)
        close(row["p_holm_primary2"], expected["holm_primary2"], 1e-12)
        for method in ("significance", "exact_isolated"):
            r = next(x for x in analysis["rows"] if x["ber"] == 16 / 1024
                     and x["budget"] == int(budget) and x["method"] == method)
            close(r["accuracy"], record["accuracy_significance_and_isolated"], 1e-12)


def finite_nll(run: Path, policy: str) -> tuple[float, float, float]:
    finite_sum = 0.0
    finite_count = failed_count = total = 0
    field_means = []
    for path in sorted((run / "final/ber_02").glob("field_???.json")):
        risk = json.loads(path.read_text())["policies"][policy]["risk"]
        losses = []
        for pair, failure in zip(risk["logits"], risk["failure_kinds"]):
            if failure is not None or pair is None or not all(map(math.isfinite, pair)):
                failed_count += 1
                losses.append(30.0)
            else:
                loss = min(30.0, max(0.0, pair[1] - pair[0]))
                finite_sum += loss
                finite_count += 1
                losses.append(loss)
        close(sum(losses) / len(losses), risk["mean_nll"], 1e-5)
        total += len(losses)
        field_means.append(risk["mean_nll"])
    assert len(field_means) == 32 and total == 32 * 256
    return failed_count / total, finite_sum / finite_count, sum(field_means) / 32


def verify_lm() -> None:
    for record in RECEIPT["language_model_pilots"]:
        run, analysis = verify_common(record)
        plan = json.loads((FROZEN_PLANS / (record["run"].removesuffix("_run1") + ".json")).read_text())
        assert plan["eligible_bits"] == 96
        assert plan["stored_precision"] == record["stored_precision"]
        assert plan["computation_precision"] == record["computation_precision"]
        assert plan["selection"]["sample_count"] == 127
        assert plan["final"]["sample_count"] == 256
        close(analysis["clean"]["perplexity"], record["clean_pilot_perplexity"], 1e-10)
        for budget, expected in record["high_ber_rows"].items():
            for method, bit_position in (("significance", 31 if record["stored_precision"] == "FP32" else 15),
                                         ("exact_isolated", 30 if record["stored_precision"] == "FP32" else 14)):
                mask = json.loads((run / "selection" / (method + ".json")).read_text())["masks"][budget]
                assert len(mask) == int(budget)
                assert all(s.endswith(f"#b{bit_position}") for s in mask)
                fail, finite, aggregate = finite_nll(run, f"{method}_b{budget}")
                close(fail, expected["isolated" if method == "exact_isolated" else method]["failed_fraction"])
                close(finite, expected["isolated" if method == "exact_isolated" else method]["finite_nll"])
                close(aggregate, expected["isolated" if method == "exact_isolated" else method]["all_nll"])
            random = next(r for r in analysis["rows"] if r["ber"] == 4 / 96
                          and r["budget"] == int(budget) and r["method"] == f"random_b{budget}")
            close(random["mean_nll"], expected["random_all_nll"])

    fp32 = SOURCE / "pythia160m_fp32_targeted96_v1_run1"
    bf16 = SOURCE / "pythia160m_bf16fp32_targeted96_v1_run1"
    position_map = {"31": "15", "30": "14", "22": "6"}
    for arm in ("ber_01", "ber_02"):
        files = sorted((fp32 / "final" / arm).glob("field_???.json"))
        assert len(files) == 32
        for path in files:
            original = json.loads(path.read_text())
            corresponding = json.loads((bf16 / "final" / arm / path.name).read_text())
            mapped_events = [
                re.sub(r"#b(31|30|22)$", lambda m: "#b" + position_map[m.group(1)], event)
                for event in original["events"]
            ]
            assert original["seed"] == corresponding["seed"]
            assert mapped_events == corresponding["events"]


if __name__ == "__main__":
    verify_distilbert()
    verify_lm()
    print("Targeted cross-precision receipts verified from finalized field records.")
