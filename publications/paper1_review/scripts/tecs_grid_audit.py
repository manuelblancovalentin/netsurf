#!/usr/bin/env python3
"""Offline full-grid audit for the TECS revision (no inference).

Reads the six finalized main-grid studies' saved policy-field records
(read-only), and writes a NEW write-once derived directory.  Nothing in a
finalized bundle is modified.

Prespecified definitions (fixed in this file before any per-cell regret or
win/loss number was computed; the author had, however, already seen the two
plotted slices of the previous manuscript, so this is not a blind analysis):

USEFUL REGION (task-specific, per (model, BER, budget, method) cell):
  classification (ResNet-8, DS-CNN, DistilBERT):
      mean accuracy >= clean accuracy - 0.05          (primary)
  ToyCar:
      mean complete-file ROC-AUC >= clean AUC - 0.05   (primary)
  language models (Pythia, Qwen):
      mean capped token NLL <= clean NLL + 0.5 nats and zero failed units
      (perplexity ratio <= e^0.5 ~ 1.65)               (primary)
  Sensitivity thresholds: tight (0.01 / 0.01 / 0.1) and loose (0.10 / 0.10 / 1.0).
A (model, BER, budget) cell is DECISION-USEFUL when at least one measured
ranking (not the zero/full controls) is useful there.

UNITS: the independent unit is the proposed fault field.  All method
contrasts are paired on the field.  Intervals are 95% two-sided Student-t on
the paired field differences; tests are two-sided paired sign-flip Monte
Carlo (20,000 draws, fixed seed) with Holm correction inside each model over
the stated family.  Everything is conditional on one frozen checkpoint, one
evaluation cohort and one constructed mask per deterministic method/budget.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

W = Path("/projects/netsurf/manuelbv/workspace/implementation-worktrees")
STUDIES = {
    "resnet8": dict(kind="tflite_cls", unit="nats/image",
        path=W / "m12-fkeras-multibench/workspace/runs/m12_resnet8_fkeras_matched_grid_v1/20261105T175030.135296Z-c799a357/measurements/trials.csv",
        eligible_bits=618_880),
    "kws": dict(kind="tflite_cls", unit="nats/clip",
        path=W / "reliability-db/workspace/runs/kws_partial_tier_fkeras_matched_v1/20261104T042124.586895Z-b0284b42/measurements/trials.csv",
        eligible_bits=176_128),
    "toycar": dict(kind="tflite_auc", unit="1-AUC",
        path=W / "m12-fkeras-multibench/workspace/runs/m12_toycar_fkeras_matched_grid_v1/20261105T210814.691588Z-16296f36/measurements/trials.csv",
        eligible_bits=2_113_536),
    "distilbert": dict(kind="w8a8_cls", unit="nats/example",
        path=W / "w8a8-ber-stress/workspace/runs/w8a8_wide_ber_matched_v1/distilbert_sst2_int8/derived/trials.csv",
        eligible_bits=344_469_504),
    "pythia": dict(kind="w8a8_lm", unit="nats/token",
        path=W / "pythia-wide-confirm/workspace/runs/pythia_w8a8_high_ber_clean_gated_v2/derived/trials.csv",
        eligible_bits=988_545_024),
    "qwen": dict(kind="w8a8_lm", unit="nats/token",
        path=W / "w8a8-ber-stress/workspace/runs/w8a8_wide_ber_matched_v1/qwen3_0_6b_int8/derived/trials.csv",
        eligible_bits=4_767_875_072),
}
METHOD_ALIASES = {
    "random_global": "random", "msb_global": "msb", "numerical_jump_global": "jump",
    "gradient_global": "global_grad", "per_example_absolute_global": "abs_grad",
    "per_example_positive_global": "pos_grad", "fkeras_paper": "fkeras",
    "random": "random", "msb_first": "msb", "numerical_bit_jump": "jump",
    "global_first_order": "global_grad", "per_example_absolute": "abs_grad",
    "per_example_positive": "pos_grad",
    "no_protection": "unprotected", "unprotected": "unprotected", "full_protection": "full",
}
RANKINGS = ["random", "msb", "jump", "global_grad", "abs_grad", "pos_grad", "fkeras"]
THRESH = {"primary": dict(acc=0.05, auc=0.05, nll=0.5),
          "tight": dict(acc=0.01, auc=0.01, nll=0.1),
          "loose": dict(acc=0.10, auc=0.10, nll=1.0)}
RISKS = {"tflite_cls": ("loss", "err"), "w8a8_cls": ("loss", "err"), "tflite_auc": ("loss",), "w8a8_lm": ("loss",)}
RISK_UNIT = {"err": "error rate (1-accuracy)"}
SIGNFLIP_DRAWS = 20_000
SEED = 20261110


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(model: str):
    """Return rows: dict(ber, method, coverage, field, loss, quality, failed, proposed, applied, suppressed)."""
    spec = STUDIES[model]
    rows = []
    with open(spec["path"], newline="") as f:
        for r in csv.DictReader(f):
            if spec["kind"].startswith("tflite"):
                ber_s, pol = r["policy"].split("/", 1)
                ber = float(ber_s.split("=")[1])
                if "@" in pol:
                    name, cov = pol.split("@"); cov = float(cov)
                else:
                    name, cov = pol, (0.0 if pol == "no_protection" else 1.0)
                field = (r["replication"], int(r["trial"]), r["schedule_sha256"])
                if spec["kind"] == "tflite_cls":
                    loss = float(r["categorical_crossentropy"]); qual = float(r["accuracy"])
                else:
                    assert r["primary_metric_name"] == "roc_auc"
                    qual = float(r["primary_metric_value"]); loss = 1.0 - qual
                failed = 0.0
                prop, app, sup = int(r["proposed_events"]), int(r["applied_events"]), int(r["suppressed_events"])
            else:
                ber = float(r["ber"]); pol = r["policy"]; cov = float(r["coverage"])
                name = pol.rsplit("_b", 1)[0] if "_b" in pol and pol.rsplit("_b", 1)[1].isdigit() else pol
                field = (int(r["field_index"]), r["field_sha256"])
                loss = float(r["mean_nll_nats"])
                qual = float(r["accuracy"]) if r["accuracy"] not in ("", None) else float("nan")
                failed = int(r["failed_units"]) / int(r["total_units"])
                prop, app, sup = int(r["proposed"]), int(r["applied"]), int(r["suppressed"])
            rows.append(dict(ber=ber, method=METHOD_ALIASES[name], coverage=round(cov, 6), field=field,
                             loss=loss, err=1.0 - qual, quality=qual, failed=failed, proposed=prop, applied=app, suppressed=sup))
    return rows


def t_ci(d: np.ndarray):
    n = len(d); m = float(np.mean(d))
    if n < 2:
        return m, float("nan"), float("nan")
    se = float(np.std(d, ddof=1)) / math.sqrt(n)
    h = stats.t.ppf(0.975, n - 1) * se
    return m, m - h, m + h


def signflip_p(d: np.ndarray, rng) -> float:
    if np.allclose(d, 0.0):
        return 1.0
    obs = abs(float(np.mean(d)))
    signs = rng.integers(0, 2, size=(SIGNFLIP_DRAWS, len(d)), dtype=np.int8) * 2 - 1
    null = np.abs((signs * d).mean(axis=1))
    return float((1 + np.sum(null >= obs - 1e-15)) / (SIGNFLIP_DRAWS + 1))


def holm(pvals):
    idx = np.argsort(pvals); m = len(pvals); out = np.empty(m); run = 0.0
    for rank, i in enumerate(idx):
        run = max(run, min(1.0, (m - rank) * pvals[i])); out[i] = run
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="new write-once output directory")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(SEED)
    provenance = {"schema": "paper1-tecs-grid-audit/2.0", "seed": SEED, "signflip_draws": SIGNFLIP_DRAWS,
                  "thresholds": THRESH, "sources": {}}

    cond_rows, regret_rows, pair_rows, harm_rows, default_rows, acct_rows = [], [], [], [], [], []
    for model, spec in STUDIES.items():
        provenance["sources"][model] = {"path": str(spec["path"]), "sha256": sha256(spec["path"])}
        rows = load(model)
        # ---- index: cell -> method -> {field: row}
        cells = defaultdict(lambda: defaultdict(dict))
        for r in rows:
            key = (r["ber"], r["coverage"])
            assert r["field"] not in cells[key][r["method"]], "duplicate policy-field"
            cells[key][r["method"]][r["field"]] = r
        bers = sorted({k[0] for k in cells}); covs = sorted({k[1] for k in cells if 0 < k[1] < 1})
        # ---- event accounting: proposed == applied + suppressed; applied <= unprotected applied on same field
        bad_acct = 0; bad_subset = 0; n_rows = 0
        for r in rows:
            n_rows += 1
            if r["proposed"] != r["applied"] + r["suppressed"]:
                bad_acct += 1
        for ber in bers:
            unp = cells[(ber, 0.0)]["unprotected"]
            for cov in covs:
                for meth, fr in cells[(ber, cov)].items():
                    for fld, r in fr.items():
                        u = unp[fld]
                        if r["proposed"] != u["proposed"] or r["applied"] > u["applied"]:
                            bad_subset += 1
        acct_rows.append(dict(model=model, policy_field_rows=n_rows, proposed_ne_applied_plus_suppressed=bad_acct,
                              protected_applied_exceeds_unprotected_or_field_mismatch=bad_subset))
        # ---- clean reference from full protection (zero applied faults)
        full = [r for r in rows if r["method"] == "full"]
        clean_loss = float(np.mean([r["loss"] for r in full])); clean_q = float(np.nanmean([r["quality"] for r in full])) if spec["kind"] != "w8a8_lm" else float("nan")
        assert max(r["applied"] for r in full) == 0
        assert np.ptp([r["loss"] for r in full]) < 1e-9, "full-protection control is not a constant clean reference"

        def useful(mean_loss, mean_q, mean_failed, level):
            t = THRESH[level]
            if spec["kind"] in ("tflite_cls", "w8a8_cls"):
                return mean_q >= clean_q - t["acc"]
            if spec["kind"] == "tflite_auc":
                return mean_q >= clean_q - t["auc"]
            return (mean_loss <= clean_loss + t["nll"]) and mean_failed == 0.0

        # ---- condition table
        cell_useful = {}
        for (ber, cov), meths in sorted(cells.items()):
            for meth, fr in meths.items():
                flds = sorted(fr)
                loss = np.array([fr[f]["loss"] for f in flds]); q = np.array([fr[f]["quality"] for f in flds])
                failed = np.array([fr[f]["failed"] for f in flds]); app = np.array([fr[f]["applied"] for f in flds])
                m, lo, hi = t_ci(loss)
                mq = float(np.nanmean(q)) if not np.all(np.isnan(q)) else float("nan")
                u = {lvl: useful(m, mq, float(failed.mean()), lvl) for lvl in THRESH}
                cell_useful[(ber, cov, meth)] = u
                cond_rows.append(dict(model=model, ber=ber, coverage=cov, method=meth, fields=len(flds),
                                      mean_loss=m, loss_ci_low=lo, loss_ci_high=hi, clean_loss=clean_loss,
                                      mean_quality=mq, min_field_quality=float(np.nanmin(q)) if not np.all(np.isnan(q)) else float("nan"),
                                      clean_quality=clean_q, mean_failed_fraction=float(failed.mean()),
                                      fields_with_any_failure=int(np.sum(failed > 0)),
                                      mean_applied_faults=float(app.mean()),
                                      expected_proposed_faults=spec["eligible_bits"] * ber,
                                      useful_primary=int(u["primary"]), useful_tight=int(u["tight"]), useful_loose=int(u["loose"]),
                                      loss_unit=spec["unit"]))
        # ---- per-cell analyses (loss always; error rate additionally for classifiers)
        for risk in RISKS[spec["kind"]]:
          unit = RISK_UNIT.get(risk, spec["unit"])
          fam_abs, fam_all = [], []
          for ber in bers:
              unp = cells[(ber, 0.0)]["unprotected"]
              for cov in covs:
                  meths = {m: fr for m, fr in cells[(ber, cov)].items() if m in RANKINGS}
                  flds = sorted(unp)
                  L = {m: np.array([fr[f][risk] for f in flds]) for m, fr in meths.items()}
                  Lu = np.array([unp[f][risk] for f in flds])
                  means = {m: float(v.mean()) for m, v in L.items()}
                  best = min(means, key=means.get)
                  dec_useful = {lvl: any(cell_useful[(ber, cov, m)][lvl] for m in meths) for lvl in THRESH}
                  unp_useful = cell_useful[(ber, 0.0, "unprotected")]["primary"]
                  # duplicate-mask detection (identical field outcomes => same protection decision)
                  dup_of = {}
                  order = [m for m in RANKINGS if m in L]
                  for i, a in enumerate(order):
                      for b in order[:i]:
                          if np.array_equal(L[a], L[b]):
                              dup_of[a] = b; break
                  for m in order:
                      d = L[m] - L[best]; mean, lo, hi = t_ci(d)
                      regret_rows.append(dict(model=model, risk=risk, ber=ber, coverage=cov, method=m, best_method=best,
                                              regret_mean=mean, regret_ci_low=lo, regret_ci_high=hi,
                                              method_mean_loss=means[m], best_mean_loss=means[best],
                                              unprotected_mean_loss=float(Lu.mean()),
                                              identical_outcomes_to=dup_of.get(m, ""),
                                              decision_useful_primary=int(dec_useful["primary"]),
                                              decision_useful_tight=int(dec_useful["tight"]),
                                              decision_useful_loose=int(dec_useful["loose"]),
                                              unprotected_useful_primary=int(unp_useful), loss_unit=unit))
                      # harm vs unprotected
                      dh = L[m] - Lu; hm, hlo, hhi = t_ci(dh); p = signflip_p(dh, rng)
                      harm_rows.append(dict(model=model, risk=risk, ber=ber, coverage=cov, method=m, mean_minus_unprotected=hm,
                                            ci_low=hlo, ci_high=hhi, p_signflip=p,
                                            decision_useful_primary=int(dec_useful["primary"]), loss_unit=unit))
                  for i, a in enumerate(order):
                      for b in order[i + 1:]:
                          d = L[a] - L[b]; mean, lo, hi = t_ci(d); p = signflip_p(d, rng)
                          rec = dict(model=model, risk=risk, ber=ber, coverage=cov, left=a, right=b, mean_left_minus_right=mean,
                                     ci_low=lo, ci_high=hi, p_signflip=p, identical_masks=int(np.array_equal(L[a], L[b])),
                                     decision_useful_primary=int(dec_useful["primary"]),
                                     decision_useful_tight=int(dec_useful["tight"]),
                                     decision_useful_loose=int(dec_useful["loose"]), loss_unit=unit)
                          pair_rows.append(rec); fam_all.append(rec)
                          if "abs_grad" in (a, b):
                              fam_abs.append(rec)
          # Holm inside model: family = all non-identical pairwise contrasts on the grid
          live = [r for r in fam_all if not r["identical_masks"]]
          adj = holm(np.array([r["p_signflip"] for r in live]))
          for r, a in zip(live, adj):
              r["p_holm_all_pairs_within_model"] = float(a)
          live_abs = [r for r in fam_abs if not r["identical_masks"]]
          adj = holm(np.array([r["p_signflip"] for r in live_abs]))
          for r, a in zip(live_abs, adj):
              r["p_holm_absgrad_family_within_model"] = float(a)
          for r in fam_all:
              r.setdefault("p_holm_all_pairs_within_model", float("nan"))
              r.setdefault("p_holm_absgrad_family_within_model", float("nan"))

    def write(name, rows):
        if not rows:
            return
        keys = list(rows[0].keys())
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(out / name, "x", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

    write("conditions_with_quality.csv", cond_rows)
    write("per_cell_regret.csv", regret_rows)
    write("pairwise_contrasts.csv", pair_rows)
    write("protection_vs_unprotected.csv", harm_rows)
    write("event_accounting.csv", acct_rows)

    # ---------- summaries ----------
    summ = {"event_accounting": acct_rows}
    # default-method test: abs_grad vs each other ranking
    by_model = defaultdict(list)
    for r in pair_rows:
        by_model[f'{r["model"]}:{r["risk"]}'].append(r)
    default = {}
    for model, rs in by_model.items():
        for region in ("all", "decision_useful_primary", "decision_useful_tight", "decision_useful_loose"):
            sel = [r for r in rs if "abs_grad" in (r["left"], r["right"]) and (region == "all" or r[region])]
            tally = defaultdict(lambda: dict(abs_better_ci=0, abs_worse_ci=0, unresolved_ci=0,
                                             abs_better_holm=0, abs_worse_holm=0, unresolved_holm=0, cells=0))
            for r in sel:
                other = r["right"] if r["left"] == "abs_grad" else r["left"]
                sign = 1.0 if r["left"] == "abs_grad" else -1.0  # d = abs - other
                lo, hi = (r["ci_low"], r["ci_high"]) if sign > 0 else (-r["ci_high"], -r["ci_low"])
                mean = sign * r["mean_left_minus_right"]
                t = tally[other]; t["cells"] += 1
                if hi < 0: t["abs_better_ci"] += 1
                elif lo > 0: t["abs_worse_ci"] += 1
                else: t["unresolved_ci"] += 1
                ph = r["p_holm_absgrad_family_within_model"]
                if ph < 0.05 and mean < 0: t["abs_better_holm"] += 1
                elif ph < 0.05 and mean > 0: t["abs_worse_holm"] += 1
                else: t["unresolved_holm"] += 1
            default.setdefault(model, {})[region] = dict(tally)
    summ["absgrad_vs_others"] = default
    # regret summary per method
    reg = {}
    rg = defaultdict(list)
    for r in regret_rows:
        rg[(f'{r["model"]}:{r["risk"]}', r["method"])].append(r)
    for (model, meth), rs in rg.items():
        for region in ("all", "decision_useful_primary", "decision_useful_tight", "decision_useful_loose"):
            sel = [r for r in rs if region == "all" or r[region]]
            if not sel:
                reg.setdefault(model, {}).setdefault(region, {})[meth] = dict(cells=0)
                continue
            vals = np.array([r["regret_mean"] for r in sel])
            worst = max(sel, key=lambda r: r["regret_mean"])
            reg.setdefault(model, {}).setdefault(region, {})[meth] = dict(
                cells=len(sel), best_or_tied_cells=int(np.sum(vals <= 1e-12)),
                cells_regret_ci_excludes_zero=int(sum(1 for r in sel if r["regret_ci_low"] > 0)),
                mean_regret=float(vals.mean()), median_regret=float(np.median(vals)), max_regret=float(vals.max()),
                max_regret_cell=dict(ber=worst["ber"], coverage=worst["coverage"], best=worst["best_method"],
                                     ci=[worst["regret_ci_low"], worst["regret_ci_high"]]))
    summ["regret_by_method"] = reg
    # decision-useful cell counts
    cu = defaultdict(lambda: defaultdict(int))
    seen = set()
    for r in regret_rows:
        if r["risk"] != "loss": continue
        k = (r["model"], r["ber"], r["coverage"])
        if k in seen: continue
        seen.add(k); cu[r["model"]]["cells"] += 1
        for lvl in ("primary", "tight", "loose"):
            cu[r["model"]][f"decision_useful_{lvl}"] += r[f"decision_useful_{lvl}"]
        cu[r["model"]]["unprotected_useful_primary"] += r["unprotected_useful_primary"]
        cu[r["model"]]["decision_useful_and_unprotected_not_useful"] += int(r["decision_useful_primary"] and not r["unprotected_useful_primary"])
    summ["cell_counts"] = {m: dict(v) for m, v in cu.items()}
    # harmful protection
    hm = defaultdict(lambda: defaultdict(lambda: dict(cells=0, worse_ci=0, better_ci=0, worse_p05=0, max_harm=0.0, max_harm_cell=None)))
    for r in harm_rows:
        t = hm[f'{r["model"]}:{r["risk"]}'][r["method"]]; t["cells"] += 1
        if r["ci_low"] > 0: t["worse_ci"] += 1
        if r["ci_high"] < 0: t["better_ci"] += 1
        if r["p_signflip"] < 0.05 and r["mean_minus_unprotected"] > 0: t["worse_p05"] += 1
        if r["mean_minus_unprotected"] > t["max_harm"]:
            t["max_harm"] = r["mean_minus_unprotected"]; t["max_harm_cell"] = [r["ber"], r["coverage"], r["ci_low"], r["ci_high"]]
    summ["protection_vs_unprotected"] = {m: {k: dict(v) for k, v in d.items()} for m, d in hm.items()}
    # all-pairs win/loss/unresolved
    wl = defaultdict(lambda: defaultdict(lambda: dict(left_better=0, right_better=0, unresolved=0, identical=0)))
    for r in pair_rows:
        for region in ("all", "decision_useful_primary"):
            if region != "all" and not r[region]:
                continue
            t = wl[(f'{r["model"]}:{r["risk"]}', region)][f'{r["left"]}|{r["right"]}']
            if r["identical_masks"]: t["identical"] += 1
            elif r["ci_high"] < 0: t["left_better"] += 1
            elif r["ci_low"] > 0: t["right_better"] += 1
            else: t["unresolved"] += 1
    summ["pairwise_ci_tally"] = {f"{m}:{reg_}": {k: dict(v) for k, v in d.items()} for (m, reg_), d in wl.items()}
    with open(out / "summary.json", "x") as f:
        json.dump(summ, f, indent=1, sort_keys=True)
    provenance["outputs"] = {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file()}
    with open(out / "provenance.json", "x") as f:
        json.dump(provenance, f, indent=1, sort_keys=True)
    for p in out.iterdir():
        os.chmod(p, 0o444)
    print("wrote", out)


if __name__ == "__main__":
    sys.exit(main())
