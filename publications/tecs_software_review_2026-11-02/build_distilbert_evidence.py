"""Offline manuscript audit of the frozen targeted FP32 archive; no inference.

Consumes the ORIGINAL archived READY snapshot, never today's mutable readiness
file. Recomputes paired field statistics and checks source, masks, schedules and
every bound run member. Writes only derived manuscript files, not source bundles.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import statistics
import zipfile

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def digest(value):
    return digest_bytes(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                   ensure_ascii=False, allow_nan=False).encode())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bound(document, key='payload_sha256', plan=None):
    require(document.get(key) == digest({k: v for k, v in document.items() if k != key}),
            f'{key} mismatch')
    if plan is not None:
        require(document.get('plan_sha256') == plan, 'run member belongs to another plan')


def paired_statistics(differences, seed, draws=10000):
    """Original frozen field-bootstrap and symmetric-null sign-flip procedure."""
    values = np.asarray(differences, dtype=float)
    require(values.ndim == 1 and len(values) > 1 and np.isfinite(values).all(),
            'paired finite field values required')
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(values), size=(draws, len(values)))
    ci = np.quantile(values[samples].mean(axis=1), [.025, .975])
    signs = rng.choice(np.array([-1., 1.]), size=(draws, len(values)))
    p = (1 + np.count_nonzero(np.abs((signs * values).mean(axis=1)) >=
                             abs(float(values.mean())))) / (draws + 1)
    return dict(mean=float(values.mean()), ci_low=float(ci[0]), ci_high=float(ci[1]),
                p_unadjusted=float(p), matched_pairs=len(values),
                favorable_fields=int(np.count_nonzero(values > 0)),
                adverse_fields=int(np.count_nonzero(values < 0)),
                median=float(np.median(values)), max_abs=float(np.max(np.abs(values))))


def holm(rows):
    previous = 0.
    for rank, i in enumerate(sorted(range(len(rows)), key=lambda i: rows[i]['p_unadjusted'])):
        previous = max(previous, min(1., (len(rows) - rank) * rows[i]['p_unadjusted']))
        rows[i]['p_holm'] = previous


def audit(pin):
    root = Path(pin['path'])
    hashes = {}
    for name, field in [('complete.zip', 'complete_sha256'), ('compact.zip', 'compact_sha256'),
                        ('checksums.json', 'checksums_sha256')]:
        hashes[name] = digest_bytes((root / name).read_bytes())
        require(hashes[name] == pin[field], f'pinned archive changed: {name}')
    for name, expected in json.loads((root / 'checksums.json').read_text()).items():
        require(digest_bytes((root / name).read_bytes()) == expected, f'export checksum: {name}')
    with zipfile.ZipFile(root / 'complete.zip') as z:
        require(z.testzip() is None, 'archive CRC failed')
        require(len(z.namelist()) == len(set(z.namelist())), 'duplicate archive member')
        members = {name: digest_bytes(z.read(name)) for name in z.namelist()}
        read = lambda name: json.loads(z.read(name))
        plan = read('plan.json'); bound(plan, 'plan_sha256')
        require(plan['plan_sha256'] == pin['plan_sha256'], 'unexpected scientific plan')
        require(members['assets/NETSURF_ASSET_READINESS.json'] ==
                plan['asset_manifest_sha256'] == pin['ready_snapshot_sha256'], 'original READY mismatch')
        ready = read('assets/NETSURF_ASSET_READINESS.json')
        target, = [x for x in ready['targets'] if x['target_id'] == plan['target']]
        require(target['status'] == 'READY', 'archived target not READY')
        require(target['reference_results']['smoke_report_sha256'] ==
                plan['clean_reference_report_sha256'], 'clean reference binding')
        for name, sha in plan['source_files_sha256'].items():
            require(members['source/' + name] == sha, f'frozen source mismatch: {name}')
        # Retain archive-member hashes; current code and global READY may legitimately evolve.
        keys = [f'classifier.weight[{word}]#b{bit}' for word in range(32) for bit in range(32)]
        require(plan['eligible_bits'] == len(keys) == 1024 and plan['candidate_bit_keys'] == keys,
                'not the declared classifier subset')
        require(digest(keys) == plan['eligible_population_sha256'], 'eligible universe hash')
        require(plan['budgets'] == [0, 128, 256, 1024] and
                plan['ber'] == [0, 1/1024, 4/1024, 16/1024], 'frozen grid changed')
        require(plan['final']['fields'] == 32 and plan['final']['sample_count'] == 256 and
                plan['selection']['sample_count'] == 32, 'study counts changed')
        require(plan['final_source_indices'] == list(range(32, 288)), 'final identities changed')
        require(not (set(plan['final_source_indices']) & set(plan['cohort_source_indices']['engineering'])),
                'engineering/final overlap')
        require(len(set(plan['selection_source_indices'])) == 32, 'duplicate selection indices')
        partition = dict(cohort_source_indices=plan['cohort_source_indices'],
                         cohort_records_sha256=plan['cohort_records_sha256'],
                         selection=plan['selection_source_indices'], final=plan['final_source_indices'],
                         selection_records_sha256=plan['selection_records_sha256'],
                         final_records_sha256=plan['final_records_sha256'])
        require(digest(partition) == plan['partitions_sha256'], 'partition binding changed')
        seal = read('run/selection/seal.json')
        require(seal['plan_sha256'] == plan['plan_sha256'], 'selection seal belongs to another plan')
        masks = {'unprotected': [], 'full': keys}; construction = {}
        for method in plan['methods']:
            name = f'run/selection/{method}.json'; selection = read(name)
            require(members[name] == seal['artifacts'][method]['sha256'], 'selection seal checksum')
            bound(selection, plan=plan['plan_sha256'])
            require(selection['method'] == method and
                    selection['bit_order_sha256'] == digest(keys), 'selection mapping changed')
            scores = selection['signed_scores']
            require(len(scores) == 1024 and all(math.isfinite(x) for x in scores), 'invalid scores')
            if method == 'significance':
                require(scores == [float(i % 32) for i in range(1024)], 'significance changed')
            if method == 'random':
                require(scores == np.random.default_rng(plan['selection']['seed']).random(1024).tolist(),
                        'random mask seed changed')
            order = sorted(range(1024), key=lambda i: (-scores[i], i))
            for budget in plan['budgets']:
                selected = [keys[i] for i in order[:budget]]
                require(sorted(selected) == selection['masks'][str(budget)], 'score-to-mask mismatch')
                if budget in (128, 256):
                    masks[f'{method}_b{budget}'] = sorted(selected, key=keys.index)
            construction[method] = {'member_sha256': members[name],
                                    'payload_sha256': selection['payload_sha256']}
        clean = read('run/final/clean.json'); bound(clean, plan=plan['plan_sha256'])
        raw, fields_by_arm = [], {}
        for arm, ber in enumerate(plan['ber'][1:], 1):
            fields_by_arm[arm] = []
            for field in range(32):
                entry = read(f'run/final/ber_{arm:02d}/field_{field:03d}.json')
                bound(entry, plan=plan['plan_sha256'])
                seed = plan['final']['seed'] + 100000 * arm + field
                require((entry['arm'], entry['ber'], entry['field'], entry['seed']) ==
                        (arm, ber, field, seed), 'field pairing identity mismatch')
                # Replay only the saved RNG schedule; no model or faulted inference.
                rng = np.random.default_rng(seed); ordinal = -1; events = []
                while True:
                    ordinal += int(rng.geometric(ber))
                    if ordinal >= len(keys): break
                    events.append('persistent:' + keys[ordinal])
                require(entry['schedule_events'] == events and entry['schedule_sha256'] == digest(events),
                        'proposed fault schedule mismatch')
                require(set(entry['policies']) == set(masks), 'missing policy or control')
                require(len(events) <= plan['max_events_per_field'], 'schedule exceeds frozen cap')
                proposed = {x.removeprefix('persistent:') for x in events}
                for policy, protected in masks.items():
                    row = entry['policies'][policy]; report = row['report']; risk = row['risk']
                    applied = len(proposed - set(protected))
                    require(row['mask_sha256'] == digest(protected), 'protected set mismatch')
                    require(report == dict(schedule_hash=digest(events), proposed_events=len(events),
                            applied_events=applied, suppressed_events=len(events)-applied,
                            unique_applied_bits=applied), 'event accounting mismatch')
                    require(row['expected_faults'] == dict(proposed=1024*ber,
                            applied=(1024-len(protected))*ber, suppressed=len(protected)*ber),
                            'expected exposure mismatch')
                    require(risk['total_units'] == 256 and math.isfinite(risk['mean_nll']) and
                            0 <= risk['mean_nll'] <= 30 and 0 <= risk['accuracy'] <= 1,
                            'invalid finite task risk')
                    if not applied:
                        require(risk == clean['risk'], 'zero-applied state does not match clean inference')
                    raw.append(dict(arm=arm, ber=ber, field=field, seed=seed,
                        schedule_sha256=entry['schedule_sha256'], policy=policy, budget=len(protected),
                        mean_nll=risk['mean_nll'], accuracy=risk['accuracy'],
                        failed_units=risk['failed_units'], **report))
                fields_by_arm[arm].append(entry)
        analysis = read('run/analysis.json'); bound(analysis, plan=plan['plan_sha256'])
        require(analysis['payload_sha256'] == pin['analysis_payload_sha256'], 'analysis identity changed')
        means = []
        for saved in analysis['rows']:
            policy = saved['method'] if saved['method'] in ('unprotected', 'full') else f"{saved['method']}_b{saved['budget']}"
            selected = [r for r in raw if r['ber'] == saved['ber'] and r['policy'] == policy]
            require(len(selected) == 32, 'pooled mean misses fields')
            for key in ('mean_nll', 'accuracy'):
                require(statistics.mean(r[key] for r in selected) == saved[key], 'saved aggregate differs')
            means.append({**saved, 'observed_mean_proposed': statistics.mean(r['proposed_events'] for r in selected),
                          'observed_mean_applied': statistics.mean(r['applied_events'] for r in selected)})
        pairs, accuracy = [], []
        for arm, ber in enumerate(plan['ber'][1:], 1):
            for budget in (128, 256):
                for baseline in ('random', 'significance'):
                    fields = fields_by_arm[arm]
                    values = [r['policies'][f'{baseline}_b{budget}']['risk']['mean_nll'] -
                              r['policies'][f'exact_isolated_b{budget}']['risk']['mean_nll'] for r in fields]
                    stat = paired_statistics(values, plan['analysis']['seed'] + len(pairs))
                    pairs.append(dict(ber=ber, budget=budget, baseline=baseline, **stat))
                    # Additional descriptive accuracy interval, not a new confirmatory family.
                    diff = [r['policies'][f'exact_isolated_b{budget}']['risk']['accuracy'] -
                            r['policies'][f'{baseline}_b{budget}']['risk']['accuracy'] for r in fields]
                    a = paired_statistics(diff, plan['analysis']['seed'] + len(pairs) - 1)
                    accuracy.append(dict(ber=ber, budget=budget, baseline=baseline,
                                         **{k: a[k] for k in ('mean', 'ci_low', 'ci_high', 'matched_pairs')}))
        holm(pairs)
        for recomputed, saved in zip(pairs, analysis['comparisons'], strict=True):
            require((recomputed['ber'], recomputed['budget'], 'exact_isolated_vs_'+recomputed['baseline']) ==
                    (saved['ber'], saved['budget'], saved['comparison']), 'comparison family changed')
            for key, old in [('mean','mean_baseline_minus_isolated'), ('p_unadjusted','p_unadjusted_signflip'),
                             ('p_holm','p_holm_family12'), ('favorable_fields','favorable_fields'),
                             ('adverse_fields','adverse_fields'), ('median','median_field_difference'),
                             ('max_abs','max_absolute_field_difference')]:
                require(math.isclose(recomputed[key], saved[old], rel_tol=1e-12, abs_tol=1e-14), 'paired statistic differs: '+key)
            require(np.allclose([recomputed['ci_low'], recomputed['ci_high']],
                                saved['ci95_percentile_bootstrap'], rtol=1e-12, atol=1e-14), 'paired interval differs')
    receipt = dict(schema_version='manuscript-distilbert-audit/v1', sources=pin,
        archive_members_sha256=members, archive_files_sha256=hashes,
        plan=plan, target=target, selection=construction, clean=clean['risk'],
        verified_fields=96, verified_policy_evaluations=len(raw),
        verified_bound_run_records=101, original_comparisons_recomputed=12,
        historical_source_writes=0, new_model_inferences=0,
        scientific_analysis_payload_sha256=analysis['payload_sha256'],
        elapsed_runtime='not recorded in the finalized scientific members; ZIP timestamps are not timers',
        statistical_scope='fixed 256 inputs and realized masks; 32 independent fields per BER; not input or mask population uncertainty')
    return receipt, raw, means, pairs, accuracy


def write_csv(path, rows):
    with path.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def effect_number(value):
    """Display all capped cross-entropy effects in nats, without row scaling."""
    return f'{value:+.2e}' if 0 < abs(value) < .01 else f'{value:+.3f}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh-draft', action='store_true')
    args = parser.parse_args()
    out = HERE / 'evidence'; out.mkdir(exist_ok=True)
    require(args.refresh_draft or not (out/'distilbert_audit.json').exists(), 'derived audit already exists')
    pin = json.loads((ROOT/'config/paper1_distilbert_source.json').read_text())
    receipt, raw, means, pairs, accuracy = audit(pin)
    (out/'distilbert_audit.json').write_text(json.dumps(receipt, indent=2, sort_keys=True)+'\n')
    for name, rows in [('fields',raw), ('means',means), ('paired_loss',pairs), ('paired_accuracy',accuracy)]:
        write_csv(out/f'distilbert_{name}.csv', rows)
    lines = []
    for row in pairs:
        lines.append(' & '.join([
            str(int(1024 * row['ber'])), f"{100 * row['budget'] / 1024:g}\\%",
            row['baseline'], effect_number(row['mean']),
            f"[{effect_number(row['ci_low'])}, {effect_number(row['ci_high'])}]",
            f"{row['p_holm']:.4f}",
        ]) + r' \\')
    (HERE/'overleaf/tables/distilbert_paired.tex').write_text('\n'.join(lines)+'\n')
    (out/'distilbert_table_provenance.json').write_text(json.dumps(dict(
        artifact='overleaf/tables/distilbert_paired.tex', source=pin,
        recipe='All 12 original paired capped-loss contrasts in nats, baseline minus isolated; 32 matched fields per BER, original 10000 field bootstrap/sign-flip draws and Holm family12. No row scaling.',
        accuracy='distilbert_paired_accuracy.csv; additional descriptive intervals, no significance test',
        new_model_inferences=0), indent=2)+'\n')
    print(json.dumps({k:receipt[k] for k in ('verified_fields','verified_policy_evaluations',
        'original_comparisons_recomputed','scientific_analysis_payload_sha256','new_model_inferences')}, indent=2))


if __name__ == '__main__':
    main()
