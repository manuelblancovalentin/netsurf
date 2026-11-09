"""Matched eight-BER W8A8 campaign reusing independently sealed ranking masks.

This is a new write-once study. Its proposed packed Bernoulli field is saved
once and reused for every policy at that target/BER/field identity. Historical
plans, masks and final observations are read-only inputs, never altered.
"""

from __future__ import annotations

import argparse
import csv
import io
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np

from .manifests import canonical_json_bytes, digest_object, sha256_file
from .w8a8_assets import INTERPRETER, load_ready_w8a8
from .w8a8_campaign import _evaluation, _risk_record, METHODS, INTERIOR
from .w8a8_dense_ber_pilot import LAW_VERSION, packed_field
from .w8a8_dense_protection import (Protection, boundary_protections,
    injected_packed, prepared_protection_bytes, verify_protection)
from .w8a8_faults import MSBMask, W8A8FaultAdapter, W8A8Universe


ROOT = Path(__file__).resolve().parent.parent
SOURCE_ROOT = Path('/projects/netsurf/manuelbv/workspace/implementation-worktrees/transformer-integration')
SOURCE_NAMES = {
    'distilbert_sst2_int8': ('distilbert_sst2_w8a8_reliability_v2',
                              'distilbert_sst2_w8a8_reliability_v2_run1'),
    'pythia160m_int8': ('pythia160m_w8a8_reliability_v3',
                        'pythia160m_w8a8_reliability_v3_run1'),
    'qwen3_0_6b_int8': ('qwen3_0_6b_w8a8_reliability_v3',
                        'qwen3_0_6b_w8a8_reliability_v3_run1'),
}
BERS = (1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
FIELDS_PER_BER = 24
SEED_BASE = 1532001
SOURCE_FILES = ('netsurf/w8a8_wide_ber_campaign.py', 'netsurf/w8a8_dense_ber_pilot.py',
                'netsurf/w8a8_dense_protection.py', 'netsurf/w8a8_faults.py',
                'netsurf/w8a8_campaign.py', 'netsurf/w8a8_rank.py',
                'netsurf/w8a8_assets.py', 'netsurf/transformer_risk.py')


def _write(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, stage = tempfile.mkstemp(prefix='.write-once-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as out:
            out.write(canonical_json_bytes(obj) + b'\n')
            out.flush()
            os.fsync(out.fileno())
        os.link(stage, path)
    finally:
        if os.path.exists(stage):
            os.unlink(stage)


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _checked(path: Path) -> dict:
    obj = _read(path)
    if obj.get('payload_sha256') != digest_object({k: v for k, v in obj.items()
                                                   if k != 'payload_sha256'}):
        raise ValueError(f'hashed source member differs: {path}')
    return obj


def _source(target: str) -> tuple[Path, Path, dict, dict]:
    plan_name, run_name = SOURCE_NAMES[target]
    plan_path = SOURCE_ROOT / 'workspace/plans' / plan_name / 'plan.json'
    run_path = SOURCE_ROOT / 'workspace/runs' / run_name
    original = _read(plan_path)
    seal = _checked(run_path / 'selection/seal.json')
    if (sha256_file(plan_path) != (plan_path.parent / 'FROZEN').read_text().strip() or
        original.get('plan_sha256') != digest_object({k: v for k, v in original.items()
                                                      if k != 'plan_sha256'}) or
        seal.get('status') != 'ALL_SELECTION_SEALED' or
        seal.get('plan_sha256') != original['plan_sha256'] or
        original.get('target') != target or tuple(original.get('methods', ())) != METHODS):
        raise ValueError('original selection plan or seal is not validated')
    if not (SOURCE_ROOT / 'workspace/runs' / f'{run_name}_export/complete.zip').is_file():
        raise ValueError('source whole-model scientific export is not finalized')
    return plan_path, run_path, original, seal


def _policy_records(bits: int, run_root: Path, source_plan: dict, seal: dict) -> list[dict]:
    budgets = {str(k): int(v) for k, v in source_plan['interior_budgets_bits'].items()}
    if {float(k) for k in budgets} != set(INTERIOR):
        raise ValueError('source protection budget grid differs')
    policies = []
    for name, protection in boundary_protections(bits).items():
        policies.append({'name': name, 'kind': protection.kind, 'budget': protection.budget,
                         'mask_digest': protection.digest})
    for method in METHODS:
        for fraction in INTERIOR:
            budget = budgets[str(fraction)]
            name = f'{method}_b{budget}'
            if method == 'msb_first':
                desc = {'name': name, 'kind': 'msb', 'budget': budget,
                        'mask_digest': MSBMask(bits, budget).digest}
                if seal['msb'][str(budget)] != desc['mask_digest']:
                    raise ValueError('source MSB mask identity changed')
            else:
                item = seal['masks'][method][str(budget)]
                desc = {'name': name, 'kind': 'packed', 'budget': budget,
                        'mask_digest': item['digest'], 'file_sha256': item['sha256'],
                        'path': str(run_root / item['path'])}
                if item['bytes'] != bits//8:
                    raise ValueError('source packed mask file size differs')
            policies.append(desc)
    if len(policies) != 26 or len({p['name'] for p in policies}) != 26:
        raise ValueError('source six-method policy family incomplete')
    return policies


def _protection(record: dict, bits: int) -> Protection:
    return Protection(record['kind'], bits, record['budget'], record['mask_digest'],
                      Path(record['path']) if record.get('path') else None,
                      record.get('file_sha256'))


def freeze(manifest_path: Path, target: str, root: Path) -> dict:
    if target not in SOURCE_NAMES or root.exists():
        raise ValueError('target unsupported or new wide-BER result already exists')
    if Path(__import__('sys').executable).resolve() != INTERPRETER.resolve():
        raise ValueError(f'freeze requires pinned W8A8 interpreter: {INTERPRETER}')
    asset = load_ready_w8a8(manifest_path, target)
    universe = W8A8Universe(asset)
    source_plan_path, source_run, source_plan, seal = _source(target)
    if (source_plan['eligible_universe']['bits'] != universe.bits or
        source_plan['eligible_universe']['sha256'] != universe.digest or
        source_plan['target_record_sha256'] != asset.record_sha256):
        raise ValueError('source mask eligible-bit population or READY asset differs')
    policies = _policy_records(universe.bits, source_run, source_plan, seal)
    for item in policies:
        verify_protection(_protection(item, universe.bits))
    seeds = [[SEED_BASE + arm*100_000 + i for i in range(FIELDS_PER_BER)]
             for arm in range(len(BERS))]
    source_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                            cwd=ROOT, text=True).strip()
    plan = {'schema': 'netsurf-w8a8-wide-ber-matched/v1', 'target': target,
            'status': 'FROZEN_BEFORE_NEW_FAULT_FIELDS', 'source_commit': source_commit,
            'source_files_sha256': {file: sha256_file(ROOT/file) for file in SOURCE_FILES},
            'ready_manifest_sha256': asset.manifest_sha256,
            'target_record_sha256': asset.record_sha256,
            'checkpoint_sha256': asset.record['model']['checkpoint_sha256']
                if 'checkpoint_sha256' in asset.record['model'] else None,
            'eligible_population_sha256': universe.digest, 'eligible_bits': universe.bits,
            'source_plan_path': str(source_plan_path),
            'source_plan_file_sha256': sha256_file(source_plan_path),
            'source_selection_seal_sha256': sha256_file(source_run/'selection/seal.json'),
            'source_plan_sha256': source_plan['plan_sha256'],
            'final_cohort': source_plan['final'],
            'selection_cohort': source_plan['selection'],
            'policies': policies, 'methods': list(METHODS),
            'budget_fractions': list(INTERIOR), 'bers': list(BERS),
            'field_seeds': seeds, 'fields_per_ber': FIELDS_PER_BER,
            'execution_shards': 4,
            'fault_law': LAW_VERSION,
            'fault_lifetime': 'persistent stored INT8 XOR across complete final cohort',
            'mitigation': 'ideal logical protected-event suppression, not physical ECC',
            'primary_metric': source_plan['primary_metric'],
            'nonfinite_penalty_nats': source_plan['nonfinite_penalty_nats'],
            'statistics': {'unit': 'independent whole-model proposed field; paired across masks',
                           'fixed_mask_warning': 'one frozen ranking mask per deterministic method and budget',
                           'primary_family': 'five non-MSB methods versus MSB at four budgets and eight BERs',
                           'multiplicity': 'Holm across 160 within-model paired contrasts',
                           'interval': 'paired field bootstrap percentile 95%, 10000 draws',
                           'test': 'two-sided paired sign-flip Monte Carlo, 10000 draws'},
            'counts': {'fields': len(BERS)*FIELDS_PER_BER, 'policy_arms_per_field': 26,
                       'actual_model_evaluations_per_field': 25,
                       'planned_faulted_model_evaluations': len(BERS)*FIELDS_PER_BER*25,
                       'packed_fault_field_bytes': len(BERS)*FIELDS_PER_BER*universe.codes,
                       'scored_units_per_evaluation': source_plan['counts']['scored_final_units_per_request']},
            'exposure_note': 'wide BER grid was chosen after a separate validation/engineering pilot; exploratory until independently confirmed'}
    plan['plan_sha256'] = digest_object(plan)
    root.mkdir(parents=True, exist_ok=False)
    _write(root/'plan.json', plan)
    _write(root/'FROZEN.json', {'plan_file_sha256': sha256_file(root/'plan.json')})
    return {'target': target, 'plan_sha256': plan['plan_sha256'], 'counts': plan['counts']}


def validate(manifest_path: Path, root: Path, *, verify_masks: bool = False) -> tuple[dict, object]:
    plan = _read(root/'plan.json')
    if (plan.get('schema') != 'netsurf-w8a8-wide-ber-matched/v1' or
        sha256_file(root/'plan.json') != _read(root/'FROZEN.json')['plan_file_sha256'] or
        plan.get('plan_sha256') != digest_object({k: v for k, v in plan.items()
                                                  if k != 'plan_sha256'}) or
        any(sha256_file(ROOT/file) != digest for file, digest in
            plan['source_files_sha256'].items())):
        raise ValueError('new wide-BER frozen plan or execution source changed')
    # Analysis/finalization never loads or executes a model. The two commands
    # that can infer (freeze cohort audit, run-shard) gate the pinned interpreter.
    asset = load_ready_w8a8(manifest_path, plan['target'], require_interpreter=False)
    universe = W8A8Universe(asset)
    source_plan_path, source_run, source_plan, seal = _source(plan['target'])
    if (plan['ready_manifest_sha256'] != asset.manifest_sha256 or
        plan['target_record_sha256'] != asset.record_sha256 or
        plan['eligible_bits'] != universe.bits or
        plan['eligible_population_sha256'] != universe.digest or
        str(source_plan_path) != plan['source_plan_path'] or
        sha256_file(source_plan_path) != plan['source_plan_file_sha256'] or
        sha256_file(source_run/'selection/seal.json') != plan['source_selection_seal_sha256'] or
        source_plan['final'] != plan['final_cohort'] or
        plan['policies'] != _policy_records(universe.bits, source_run, source_plan, seal)):
        raise ValueError('READY model, source cohort or sealed masks differ from frozen plan')
    if verify_masks:
        for record in plan['policies']:
            verify_protection(_protection(record, universe.bits))
    return plan, asset


def _field_paths(root: Path, arm: int, index: int) -> tuple[Path, Path, Path]:
    directory = root/'fields'/f'arm_{arm:02d}'/f'field_{index:03d}'
    return directory/'proposed.u8', directory/'field.json', directory


def _field(root: Path, plan: dict, arm: int, index: int) -> tuple[Path, dict, Path]:
    path, meta_path, directory = _field_paths(root, arm, index)
    directory.mkdir(parents=True, exist_ok=True)
    ber, seed = plan['bers'][arm], plan['field_seeds'][arm][index]
    if path.exists() or meta_path.exists():
        if not path.exists() or not meta_path.exists():
            raise RuntimeError('partial proposed field requires manual audit; not overwriting')
        meta = _read(meta_path)
        if (meta.get('seed') != seed or meta.get('ber') != ber or
            meta.get('universe_bits') != plan['eligible_bits'] or
            meta.get('law_version') != LAW_VERSION or
            meta.get('bytes') != path.stat().st_size or
            meta.get('sha256') != sha256_file(path)):
            raise ValueError('saved proposed field differs from frozen schedule')
    else:
        meta = packed_field(path, plan['eligible_bits'], ber, seed)
        _write(meta_path, meta)
    return path, meta, directory


def run_shard(manifest_path: Path, root: Path, shard: int, shards: int) -> dict:
    if Path(__import__('sys').executable).resolve() != INTERPRETER.resolve():
        raise ValueError(f'matched evaluation requires pinned W8A8 interpreter: {INTERPRETER}')
    if not 1 <= shards <= 16 or not 0 <= shard < shards:
        raise ValueError('invalid deterministic field shard')
    plan, asset = validate(manifest_path, root, verify_masks=True)
    if shards != plan['execution_shards']:
        raise ValueError('worker count differs from frozen execution plan')
    adapter = W8A8FaultAdapter(asset)
    adapter.load()
    evaluate = _evaluation(adapter, {'target': plan['target'], 'final': plan['final_cohort']})
    clean = _risk_record(evaluate())
    clean_path = root/'clean'/f'shard_{shard:02d}.json'
    if clean_path.exists():
        if _read(clean_path) != clean:
            raise ValueError('clean authoritative inference differs on resumed shard')
    else:
        _write(clean_path, clean)
    policies = [(record, _protection(record, plan['eligible_bits']))
                for record in plan['policies']]
    protected_bytes = {record['name']: prepared_protection_bytes(protection)
                       for record, protection in policies}
    completed_fields = 0
    # Interleave BER arms so every regime receives measurements early.
    for index in range(shard, FIELDS_PER_BER, shards):
        for arm in range(len(BERS)):
            field_path, meta, directory = _field(root, plan, arm, index)
            proposed_bytes = np.memmap(field_path, dtype=np.uint8, mode='r')
            for record, protection in policies:
                name = record['name']
                result_path = directory/f'{name}.json'
                if result_path.exists():
                    result = _read(result_path)
                    if (result.get('payload_sha256') != digest_object({k: v for k, v in result.items()
                                                                      if k != 'payload_sha256'}) or
                        result.get('field_sha256') != meta['sha256'] or
                        result.get('mask_sha256') != protection.digest or
                        result.get('plan_sha256') != plan['plan_sha256'] or
                        result.get('events', {}).get('proposed') != meta['proposed']):
                        raise ValueError('saved policy observation differs on resume')
                    continue
                started = time.monotonic()
                if protection.kind == 'full':
                    events = {'field_sha256': meta['sha256'], 'mask_sha256': protection.digest,
                              'proposed': meta['proposed'], 'applied': 0,
                              'suppressed': meta['proposed']}
                    risk = clean
                    origin = 'derived_exact_zero_applied_faults'
                else:
                    with injected_packed(adapter, proposed_bytes, meta, protection,
                                         protected_bytes[name]) as events:
                        risk = _risk_record(evaluate())
                    origin = 'authoritative_faulted_inference'
                if events['proposed'] != events['applied'] + events['suppressed']:
                    raise ValueError('protected-event accounting does not close')
                observation = {'plan_sha256': plan['plan_sha256'],
                    'arm': arm, 'field_index': index, 'ber': plan['bers'][arm],
                    'field_seed': meta['seed'], 'field_sha256': meta['sha256'],
                    'policy': name, 'mask_sha256': protection.digest,
                    'budget_bits': protection.budget, 'events': events,
                    'risk': risk, 'origin': origin,
                    'elapsed_seconds': time.monotonic()-started}
                observation['payload_sha256'] = digest_object(observation)
                _write(result_path, observation)
            del proposed_bytes
            completed_fields += 1
            print(json.dumps({'target': plan['target'], 'shard': shard, 'shards': shards,
                              'ber': plan['bers'][arm], 'field': index,
                              'completed_fields_this_shard': completed_fields,
                              'policies': len(policies)}), flush=True)
    clean_end = _risk_record(evaluate())
    if clean_end != clean:
        raise RuntimeError('pinned model did not restore to exact clean output after shard')
    end_path = root/'clean'/f'shard_{shard:02d}_end.json'
    if end_path.exists():
        if _read(end_path) != clean_end:
            raise ValueError('clean end differs on resumed shard')
    else:
        _write(end_path, clean_end)
    return {'target': plan['target'], 'shard': shard, 'completed_fields_this_shard': completed_fields}


def status(root: Path) -> dict:
    plan = _read(root/'plan.json')
    complete = 0
    observations = 0
    for arm in range(len(plan['bers'])):
        for index in range(FIELDS_PER_BER):
            _, meta, directory = _field_paths(root, arm, index)
            if not meta.exists():
                continue
            count = sum((directory/f"{p['name']}.json").exists() for p in plan['policies'])
            observations += count
            complete += count == len(plan['policies'])
    return {'target': plan['target'], 'complete_fields': complete,
            'planned_fields': plan['counts']['fields'],
            'saved_policy_observations': observations,
            'planned_policy_observations': plan['counts']['fields']*len(plan['policies'])}


def _csv_write_once(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    data = buffer.getvalue().encode()
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f'existing derived analysis differs: {path}')
        return
    with path.open('xb') as out:
        out.write(data)
        out.flush()
        os.fsync(out.fileno())


def _json_write_or_verify(path: Path, obj: dict) -> None:
    data = canonical_json_bytes(obj) + b'\n'
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f'existing derived analysis differs: {path}')
        return
    _write(path, obj)


def _holm(rows: list[dict], key: str = 'p_signflip') -> None:
    order = sorted(range(len(rows)), key=lambda i: rows[i][key])
    previous = 0.0
    for position, index in enumerate(order):
        previous = max(previous, min(1.0, rows[index][key]*(len(rows)-position)))
        rows[index]['p_holm_160'] = previous


def finalize(manifest_path: Path, root: Path) -> dict:
    """Offline finalization; cannot load a model or generate a proposed field."""
    if (root/'FINALIZED').exists():
        raise ValueError('wide-BER scientific result already finalized')
    plan, _ = validate(manifest_path, root, verify_masks=True)
    if status(root)['complete_fields'] != plan['counts']['fields']:
        raise ValueError('not every planned BER/field/policy observation was saved')
    clean = []
    for shard in range(plan['execution_shards']):
        start = _read(root/'clean'/f'shard_{shard:02d}.json')
        end = _read(root/'clean'/f'shard_{shard:02d}_end.json')
        if start != end:
            raise ValueError('shard did not restore original clean inference')
        clean.append(start)
    if any(item != clean[0] for item in clean):
        raise ValueError('clean authoritative inference differs across workers')
    raw_rows = []
    conditions = []
    comparisons = []
    by_arm_policy: dict[tuple[int, str], list[dict]] = {}
    by_field: dict[tuple[int, int], dict[str, dict]] = {}
    for arm, ber in enumerate(plan['bers']):
        for index in range(FIELDS_PER_BER):
            field_path, meta_path, directory = _field_paths(root, arm, index)
            meta = _read(meta_path)
            if (meta['ber'] != ber or meta['seed'] != plan['field_seeds'][arm][index] or
                meta['universe_bits'] != plan['eligible_bits'] or
                meta['law_version'] != LAW_VERSION or
                field_path.stat().st_size != plan['eligible_bits']//8 or
                sha256_file(field_path) != meta['sha256']):
                raise ValueError('saved paired proposed field differs from frozen plan')
            by_field[(arm, index)] = {}
            for policy in plan['policies']:
                item = _read(directory/f"{policy['name']}.json")
                event = item['events']
                risk = item['risk']
                if (item.get('payload_sha256') != digest_object({k: v for k, v in item.items()
                                                                if k != 'payload_sha256'}) or
                    item['plan_sha256'] != plan['plan_sha256'] or item['arm'] != arm or
                    item['field_index'] != index or item['ber'] != ber or
                    item['field_seed'] != meta['seed'] or
                    item['field_sha256'] != meta['sha256'] or
                    item['policy'] != policy['name'] or
                    item['mask_sha256'] != policy['mask_digest'] or
                    item['budget_bits'] != policy['budget'] or
                    event['field_sha256'] != meta['sha256'] or
                    event['mask_sha256'] != policy['mask_digest'] or
                    event['proposed'] != meta['proposed'] or
                    event['proposed'] != event['applied'] + event['suppressed'] or
                    risk['total_units'] != plan['counts']['scored_units_per_evaluation'] or
                    not math.isfinite(risk['metrics']['mean_nll'])):
                    raise ValueError('saved policy/field measurement violates scientific plan')
                if policy['kind'] == 'full' and (event['applied'] != 0 or risk != clean[0]):
                    raise ValueError('full protection does not equal the clean control')
                if policy['kind'] == 'none' and event['suppressed'] != 0:
                    raise ValueError('unprotected policy suppressed proposed faults')
                row = {'target': plan['target'], 'ber': ber, 'arm': arm,
                       'field_index': index, 'field_seed': meta['seed'],
                       'field_sha256': meta['sha256'], 'policy': policy['name'],
                       'budget_bits': policy['budget'],
                       'coverage': policy['budget']/plan['eligible_bits'],
                       'mask_sha256': policy['mask_digest'],
                       'proposed': event['proposed'], 'applied': event['applied'],
                       'suppressed': event['suppressed'],
                       'mean_nll_nats': risk['metrics']['mean_nll'],
                       'accuracy': risk['metrics'].get('accuracy'),
                       'failed_units': risk['failed_units'],
                       'total_units': risk['total_units'],
                       'origin': item['origin']}
                raw_rows.append(row)
                by_arm_policy.setdefault((arm, policy['name']), []).append(row)
                by_field[(arm, index)][policy['name']] = row
    for arm, ber in enumerate(plan['bers']):
        for policy in plan['policies']:
            rows = by_arm_policy[(arm, policy['name'])]
            values = np.asarray([r['mean_nll_nats'] for r in rows], dtype=np.float64)
            seed = 1532901 + arm*100 + plan['policies'].index(policy)
            boot = values[np.random.default_rng(seed).integers(
                0, len(values), size=(10000, len(values)))].mean(axis=1)
            conditions.append({'target': plan['target'], 'ber': ber,
                'policy': policy['name'], 'coverage': policy['budget']/plan['eligible_bits'],
                'budget_bits': policy['budget'], 'field_count': len(rows),
                'mean_nll_nats': float(values.mean()),
                'loss_std_nats': float(values.std(ddof=1)),
                'loss_ci95_low': float(np.quantile(boot, .025)),
                'loss_ci95_high': float(np.quantile(boot, .975)),
                'clean_nll_nats': clean[0]['metrics']['mean_nll'],
                'mean_loss_degradation_nats': float(values.mean()-clean[0]['metrics']['mean_nll']),
                'mean_accuracy': (float(np.mean([r['accuracy'] for r in rows]))
                                  if rows[0]['accuracy'] is not None else None),
                'failed_unit_count': sum(r['failed_units'] for r in rows),
                'mask_sha256': policy['mask_digest']})
        for fraction in INTERIOR:
            budget = int(plan['eligible_bits']*fraction)
            reference = f'msb_first_b{budget}'
            for method in METHODS:
                if method == 'msb_first':
                    continue
                compared = f'{method}_b{budget}'
                diff = np.asarray([by_field[(arm, i)][compared]['mean_nll_nats'] -
                                   by_field[(arm, i)][reference]['mean_nll_nats']
                                   for i in range(FIELDS_PER_BER)], dtype=np.float64)
                seed = 1533901 + len(comparisons)
                rng = np.random.default_rng(seed)
                boot = diff[rng.integers(0, len(diff), size=(10000, len(diff)))].mean(axis=1)
                signs = rng.choice((-1., 1.), size=(10000, len(diff)))
                observed = float(diff.mean())
                permuted = (diff*signs).mean(axis=1)
                p = (1+int(np.sum(np.abs(permuted) >= abs(observed))))/10001
                comparisons.append({'target': plan['target'], 'ber': ber,
                    'coverage': fraction, 'budget_bits': budget,
                    'compared_method': method, 'reference_method': 'msb_first',
                    'field_count': len(diff), 'mean_nll_difference_nats': observed,
                    'ci95_low': float(np.quantile(boot, .025)),
                    'ci95_high': float(np.quantile(boot, .975)),
                    'p_signflip': p, 'favorable_fields': int(np.sum(diff < 0)),
                    'adverse_fields': int(np.sum(diff > 0)),
                    'tied_fields': int(np.sum(diff == 0))})
    if len(conditions) != 208 or len(comparisons) != 160:
        raise ValueError('wide-BER aggregate grid incomplete')
    _holm(comparisons)
    _csv_write_once(root/'derived/trials.csv', raw_rows, list(raw_rows[0]))
    _csv_write_once(root/'derived/conditions.csv', conditions, list(conditions[0]))
    _csv_write_once(root/'derived/paired_comparisons.csv', comparisons, list(comparisons[0]))
    analysis = {'schema': 'netsurf-w8a8-wide-ber-analysis/v1',
                'plan_sha256': plan['plan_sha256'], 'target': plan['target'],
                'field_count': plan['counts']['fields'], 'trial_observations': len(raw_rows),
                'condition_rows': len(conditions), 'paired_comparisons': len(comparisons),
                'clean_risk': clean[0], 'primary_metric': plan['primary_metric'],
                'comparisons_family': plan['statistics']['primary_family'],
                'multiplicity': plan['statistics']['multiplicity'],
                'interpretation': plan['exposure_note']}
    _json_write_or_verify(root/'derived/analysis.json', analysis)
    checksums = {str(path.relative_to(root)): sha256_file(path)
                 for path in sorted(root.rglob('*')) if path.is_file() and
                 path.name not in {'manifest.json', 'FINALIZED'}}
    summary = {'schema': 'netsurf-w8a8-wide-ber-finalized/v1',
               'status': 'FINALIZED_MEASURED', 'target': plan['target'],
               'plan_sha256': plan['plan_sha256'],
               'completed_utc': datetime.now(timezone.utc).isoformat(),
               'scientific_file_sha256': checksums,
               'analysis_sha256': sha256_file(root/'derived/analysis.json')}
    summary['payload_sha256'] = digest_object(summary)
    _write(root/'manifest.json', summary)
    _write(root/'FINALIZED', {'manifest_sha256': sha256_file(root/'manifest.json')})
    return {'target': plan['target'], 'status': summary['status'],
            'fields': plan['counts']['fields'], 'conditions': len(conditions),
            'manifest_sha256': sha256_file(root/'manifest.json')}


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    fr = sub.add_parser('freeze'); fr.add_argument('manifest', type=Path)
    fr.add_argument('target', choices=SOURCE_NAMES); fr.add_argument('root', type=Path)
    run = sub.add_parser('run-shard'); run.add_argument('manifest', type=Path)
    run.add_argument('root', type=Path); run.add_argument('shard', type=int)
    run.add_argument('shards', type=int)
    st = sub.add_parser('status'); st.add_argument('root', type=Path)
    fin = sub.add_parser('finalize'); fin.add_argument('manifest', type=Path)
    fin.add_argument('root', type=Path)
    args = parser.parse_args()
    if args.command == 'freeze':
        result = freeze(args.manifest, args.target, args.root)
    elif args.command == 'run-shard':
        result = run_shard(args.manifest, args.root, args.shard, args.shards)
    elif args.command == 'finalize':
        result = finalize(args.manifest, args.root)
    else:
        result = status(args.root)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
