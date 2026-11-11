"""Offline, write-once FKeras join against the frozen 320-candidate audit.

Bound: three small empirical bundles and three saved FKeras summaries, 320
identities/model, four saved score realizations, 1000 stratified resamples.
No model, checkpoint, dataset, inference engine, or training code is loaded.
Expected resources: one CPU, under 1 GiB RAM, a few minutes; JSON output only.

Run through the existing proj netsurf Python 3.11 environment. The default
destination is data/fkeras_candidate_audit_v1.json; --check recomputes and
checks it without writing. --self-test checks the rank/tie implementation.
The original MSB point estimate AND bootstrap endpoints must be reproduced
within 1e-12 before any FKeras result is accepted. KWS/ToyCar remain blocked
when old actual calibration-input identity cannot be established.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import rankdata, spearmanr


REPO = Path('/projects/netsurf/manuelbv')
BASE = REPO / 'workspace/implementation-worktrees'
PAPER = Path(__file__).resolve().parents[1]
OUTPUT = PAPER / 'data/fkeras_candidate_audit_v1.json'
SOURCES = {
    'resnet8': (
        REPO / 'workspace/runs/m91_resnet8_estimator_validation_stratified320_v1/20261029T014136.049835Z-c23f1d77',
        BASE / 'm12-fkeras-multibench/workspace/runs/m12_resnet8_fkeras_matched_grid_v1/20261105T175030.135296Z-c799a357',
    ),
    'kws': (
        REPO / 'workspace/runs/m93_kws_estimator_validation_stratified320_v1/20261029T172802.081387Z-f3ebf3fe',
        BASE / 'reliability-db/workspace/runs/kws_partial_tier_fkeras_matched_v1/20261104T042124.586895Z-b0284b42',
    ),
    'toycar': (
        REPO / 'workspace/runs/m93_toycar_estimator_validation_stratified320_v1/20261029T181531.665041Z-bbb64f74',
        BASE / 'm12-fkeras-multibench/workspace/runs/m12_toycar_fkeras_matched_grid_v1/20261105T210814.691588Z-16296f36',
    ),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


class Bundle:
    """Read every consumed byte and check its manifest hash before parsing."""

    def __init__(self, root):
        self.root = root
        self.receipts = []
        self.manifest = json.loads(self.read('manifest.json', manifest_bound=False))
        require(self.manifest['status'] == 'complete', f'{root}: incomplete manifest')
        self.outputs = {r['path']: r for r in self.manifest['outputs']}
        marker = self.read('FINALIZED', manifest_bound=False).decode().strip()
        require(marker == 'scientific_payload_sha256=' + self.manifest['scientific_payload_sha256'],
                f'{root}: finalization marker disagrees with manifest')

    def read(self, name, manifest_bound=True):
        data = (self.root / name).read_bytes()
        digest = sha(data)
        if manifest_bound:
            expected = self.outputs[name]
            require(digest == expected['sha256'], f'{self.root / name}: hash mismatch')
            require(len(data) == expected['bytes'], f'{self.root / name}: length mismatch')
        self.receipts.append({'path': str(self.root / name), 'sha256': digest,
                              'bytes': len(data), 'manifest_bound': manifest_bound})
        return data

    def json(self, name):
        return json.loads(self.read(name))


def identity(row):
    return row['parameter'], int(row['flat_index']), int(row['bit_index'])


def strict_codes(bits, magnitudes):
    """Exact lexicographic ordering, preserving identical (bit, |H'|) ties."""
    pairs = [(int(b), float(s)) for b, s in zip(bits, magnitudes, strict=True)]
    levels = {pair: i for i, pair in enumerate(sorted(set(pairs)))}
    return np.asarray([levels[pair] for pair in pairs], dtype=np.int64)


def spearman(a, b):
    # Equivalent to the repository's analysis_m3.spearman, with average ties.
    return float(np.corrcoef(rankdata(a, method='average'), rankdata(b, method='average'))[0, 1])


def self_test():
    bits = [7, 7, 6, 7, 0]
    magnitudes = [2., 2., 999., 3., 1e9]
    codes = strict_codes(bits, magnitudes)
    require(codes.tolist() == [2, 2, 1, 3, 0], 'tier precedence or natural ties failed')
    for indices in ([0, 1, 2, 3, 4], [0, 0, 3, 2, 2]):
        a = codes[indices]
        b = np.asarray([0., 0., 2., 4., 1.])[indices]
        require(abs(spearman(a, b) - float(spearmanr(a, b).statistic)) < 1e-14,
                'average-tie correlation disagrees with independent scipy implementation')
    require(spearman(codes, codes) == 1., 'correlation orientation failed')


def bootstrap_draws(rows, seed, count):
    strata = {}
    for i, row in enumerate(rows):
        quantile = int(row['score_quantile']) if row['score_quantile'] else None
        key = (row['layer'], row['parameter_class'], int(row['bit_index']), quantile)
        strata.setdefault(key, []).append(i)
    require(len(strata) == 80 and {len(v) for v in strata.values()} == {4},
            'unexpected original candidate design')
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(count):
        draw = []
        for key in sorted(strata):
            values = strata[key]
            selected = rng.integers(0, len(values), size=len(values))
            draw.extend(values[int(i)] for i in selected)
        draws.append(np.asarray(draw, dtype=np.int64))
    return draws


def statistic(scores, damage, draws, confidence):
    values = [spearman(scores[draw], damage[draw]) for draw in draws]
    alpha = (1 - confidence) / 2
    require(all(math.isfinite(v) for v in values), 'undefined bootstrap correlation')
    return {'spearman': spearman(scores, damage),
            'ci95_low': float(np.nanquantile(values, alpha)),
            'ci95_high': float(np.nanquantile(values, 1 - alpha))}


def model_audit(model, empirical_root, fkeras_root):
    old, new = Bundle(empirical_root), Bundle(fkeras_root)
    old_summary = old.json('summary.json')
    old_payload = old.json('scientific_payload.json')
    old_config = old.json('resolved_config.json')
    old_parts = old.json('artifacts/sample_partitions.json')
    selection = old.json('artifacts/candidate_selection.json')
    rows = list(csv.DictReader(io.StringIO(old.read('measurements/empirical_bits.csv').decode())))
    new_summary = new.json('summary.json')
    new_parts = new.json('artifacts/sample_partitions.json')
    require(len(rows) == len(set(map(identity, rows))) == 320, 'candidate identities not unique/320')
    require(set(map(identity, rows)) == set(map(identity, selection)), 'selection and empirical join mismatch')
    require(new_summary['empirical_candidate_count'] == 0, 'unexpected FKeras isolated experiment')
    old_b, new_b = old.manifest['benchmark'], new.manifest['benchmark']
    identity_fields = ('id', 'adapter', 'checkpoint_sha256', 'inference_backend',
                       'injectable_parameters', 'quantization', 'quantization_backend', 'ranking_samples')
    for field in identity_fields:
        require(old_b[field] == new_b[field], f'{model}: mismatched {field}')
    require(old_payload['benchmark']['partitions'] == old_parts, 'old payload partitions differ')
    require(old_b['sample_partitions'] == old_parts, 'old manifest partitions differ')
    require(all(row['evaluation_sample_sha256'] == old_parts['empirical_validation']['sample_sha256']
                for row in rows), 'empirical loss sample hash mismatch')
    for row in rows:
        require(float(row['positive_loss_damage']) == max(0., float(row['signed_loss_difference'])),
                'positive damage definition mismatch')
        require(abs(float(row['signed_loss_difference']) -
                    (float(row['faulted_loss']) - float(row['clean_loss']))) < 1e-12,
                'saved mean loss difference mismatch')
    if model == 'toycar':
        require({r['loss_metric_name'] for r in rows} == {'reconstruction_mse'}, 'ToyCar endpoint mismatch')
    old_lock = old_payload['benchmark']['representation']['asset_lock']
    old_dataset = old_b.get('dataset_identity')
    new_dataset = new_b.get('dataset_identity')
    if old_dataset:
        require({k: v for k, v in old_dataset.items() if k != 'asset_lock_digest'} ==
                {k: v for k, v in new_dataset.items() if k != 'asset_lock_digest'},
                'dataset/preprocessing/index content declarations disagree')
    tensors = {t['parameter']: t for t in old_payload['benchmark']['representation']['tensors']}
    bits = np.asarray([int(r['bit_index']) for r in rows])
    damage = np.asarray([float(r['positive_loss_damage']) for r in rows])
    saved_msb = next(r for r in old_summary['ranking_metrics_overall'] if r['estimator'] == 's2_msb_first')
    seed = old_config['execution']['permutation_seed']
    count = saved_msb['bootstrap_replicates']
    confidence = old_config['execution']['confidence']
    require(count == 1000 and confidence == .95, 'unexpected bootstrap specification')
    draws = bootstrap_draws(rows, seed, count)
    msb = statistic(bits, damage, draws, confidence)
    for derived_key, source_key in [('spearman', 'spearman_positive_damage'),
                                    ('ci95_low', 'spearman_positive_damage_ci_low'),
                                    ('ci95_high', 'spearman_positive_damage_ci_high')]:
        require(abs(msb[derived_key] - saved_msb[source_key]) < 1e-12,
                f'{model}: failed original MSB {derived_key} reproduction')
    replicas = {}
    baseline = None
    baseline_codes = None
    for replica, diagnostics in sorted(new_summary['fkeras_diagnostics'].items()):
        require(new_parts[replica]['ranking'] == old_parts['ranking'],
                f'{model}/{replica}: label-bound ranking partition mismatch')
        metadata = diagnostics['fkeras_paper']
        require(metadata['expression'] == 'abs(sum_i lambda_i*(v_i.T theta)*v_i)', 'wrong FKeras expression')
        require(metadata['ordering_requirement'] == 'bit_significance_desc, estimator_score_desc, stable_identity',
                'wrong FKeras allocation semantics')
        mapping = {m['parameter']: m for m in metadata['parameter_mapping']}
        raw = np.asarray(metadata['raw_parameter_score'], dtype=np.float64)
        require(np.isfinite(raw).all(), 'nonfinite FKeras scores')
        for parameter, mapped in mapping.items():
            require({k: v for k, v in mapped.items() if k not in ('flat_offset', 'flat_size')} == tensors[parameter],
                    f'{model}: tensor/quantization identity mismatch for {parameter}')
            require(mapped['flat_size'] == math.prod(mapped['shape']), 'bad mapping size')
        require(sum(m['flat_size'] for m in mapping.values()) == len(raw), 'incomplete raw score map')
        candidate_scores = []
        for row in rows:
            mapped = mapping[row['parameter']]
            index, bit = int(row['flat_index']), int(row['bit_index'])
            require(0 <= index < mapped['flat_size'] and 0 <= bit < mapped['bit_width'], 'candidate out of bounds')
            candidate_scores.append(abs(float(raw[mapped['flat_offset'] + index])))
        candidate_scores = np.asarray(candidate_scores)
        codes = strict_codes(bits, candidate_scores)
        if baseline is None:
            baseline, baseline_codes = candidate_scores, codes
        replicas[replica] = {
            'joined_candidates': len(candidate_scores),
            'candidate_score_sha256_float64_le': sha(candidate_scores.astype('<f8').tobytes()),
            'exact_score_equal_to_replication_0': bool(np.array_equal(candidate_scores, baseline)),
            'max_absolute_score_difference_from_replication_0': float(np.max(np.abs(candidate_scores - baseline))),
            'strict_tier_candidate_order_and_ties_equal_to_replication_0': bool(np.array_equal(codes, baseline_codes)),
            'raw_candidate_order_and_ties_equal_to_replication_0': bool(np.array_equal(rankdata(candidate_scores), rankdata(baseline))),
            'eigensolver_seed': metadata['seed'],
        }
    require(len(replicas) == 4, 'expected four recorded FKeras score realizations')
    # Source/indices/labels alone do not identify preprocessed input arrays.
    # CIFAR's complete raw archive hash plus fixed preprocessing does; old audio
    # bundles lack raw input content hashes and have changed aggregate locks.
    input_identity_verified = model == 'resnet8'
    new_lock = None
    if input_identity_verified:
        new_payload = new.json('scientific_payload.json')
        new_execution = new_payload['replications'][0]['benchmark']
        new_lock = new_execution['representation']['asset_lock']
        require(old_b['dataset_sha256'] == new_b['dataset_sha256'] == old_payload['benchmark']['dataset_sha256'],
                'CIFAR archive identity mismatch')
        require(new_execution['dataset_sha256'] == old_b['dataset_sha256'], 'CIFAR execution archive mismatch')
        require(old_lock['preprocessing_id'] == 'mlperf-tiny/ic01/uint8-minus-128/1', 'unexpected CIFAR preprocessing')
        require({k: v for k, v in old_lock.items() if k != 'digest'} ==
                {k: v for k, v in new_lock.items() if k != 'digest'},
                'CIFAR frozen preprocessing/reference content differs')
        require(new_execution['representation']['tensors'] == old_payload['benchmark']['representation']['tensors'],
                'CIFAR frozen tensor representations differ')
        del new_payload
    record = {
        'model': model,
        'status': 'validated' if input_identity_verified else 'blocked_calibration_input_identity',
        'endpoint': 'positive reconstruction-MSE damage' if model == 'toycar' else 'positive cross-entropy damage',
        'endpoint_column': 'positive_loss_damage',
        'candidate_count': 320,
        'join': {'all_candidate_identities_unique': True, 'selection_matches_measurements': True,
                 'all_candidates_map_to_saved_scores': True, 'kernel_shapes_quantization_and_names_equal': True,
                 'ranking_partitions_including_label_bound_hash_equal': True,
                 'model_and_inference_identity_equal': True,
                 'actual_calibration_input_identity_verified': input_identity_verified},
        'ranking_partition': old_parts['ranking'],
        'empirical_partition': old_parts['empirical_validation'],
        'model_identity': {key: old_b[key] for key in identity_fields},
        'dataset_identity': {'empirical': old_dataset or {'dataset_sha256': old_b['dataset_sha256']},
                             'fkeras': new_dataset or {'dataset_sha256': new_b['dataset_sha256']}},
        'aggregate_asset_lock_digests': {
            'empirical': old_lock['digest'],
            'fkeras': new_lock['digest'] if new_lock else new_dataset['asset_lock_digest'],
            'interpretation': 'distinct lock artifacts; no field-by-field historical lock diff or path-only claim',
        },
        'original_msb_reproduction': msb,
        'bootstrap': {'seed': seed, 'replicates': count, 'confidence': confidence, 'strata': 80,
                      'candidates_per_stratum': 4, 'design': saved_msb['bootstrap_design'],
                      'group_key': ['layer', 'parameter_class', 'bit_index', 'score_quantile'],
                      'within_stratum_order': 'original empirical CSV row order',
                      'generator': 'numpy.random.default_rng; sorted strata; integers(0,n,size=n)',
                      'source_numpy_version': old.manifest['runtime']['numpy'],
                      'original_msb_point_and_interval_reproduced_tolerance': 1e-12},
        'saved_score_replication_checks': replicas,
        'score_realization_used': 'replication_0; chosen by canonical index, not observed correlation',
        'source_manifests': {'empirical': old.manifest['source'], 'fkeras': new.manifest['source']},
        'source_files': old.receipts + new.receipts,
        'results': None,
    }
    if input_identity_verified:
        record['results'] = {
            'fkeras_strict_bit_tier_ranking': statistic(baseline_codes, damage, draws, confidence),
            'fkeras_raw_absolute_projection_diagnostic': statistic(baseline, damage, draws, confidence),
        }
        record['input_identity_basis'] = 'identical complete CIFAR archive SHA-256, exact ranking indices and label-bound hash, fixed uint8-minus-128 preprocessing'
    else:
        record['blocker'] = ('Historical sample_sha256 hashes source_split, indices and labels, not actual inputs. '
                             'The aggregate asset-lock digest changed; the complete old lock and old encoded '
                             'calibration-input hash are unavailable. Matching preprocessing, official sample-index '
                             'hash and label-bound calibration partition do not prove audio-input byte identity. '
                             'No FKeras correlation is published for this model.')
    print(model, record['status'], record['results'], flush=True)
    return record


def derive():
    require(platform.python_version_tuple()[:2] == ('3', '11'), 'use prescribed Python 3.11 environment')
    self_test()
    models = [model_audit(model, *paths) for model, paths in SOURCES.items()]
    source_files = [Path(__file__).resolve(), PAPER.parents[1] / 'netsurf/samples.py',
                    PAPER.parents[1] / 'netsurf/analysis_m3.py', PAPER.parents[1] / 'netsurf/study_engine.py',
                    BASE / 'm12-fkeras-multibench/netsurf/fkeras_estimator.py',
                    BASE / 'm12-fkeras-multibench/netsurf/fkeras_baseline.py',
                    BASE / 'm12-fkeras-multibench/netsurf/ordering.py']
    implementation = []
    for path in source_files:
        data = path.read_bytes()
        implementation.append({'path': str(path), 'bytes': len(data), 'sha256': sha(data)})
    content = {
        'schema': 'netsurf.paper1.fkeras-candidate-audit/v1',
        'status': 'FINALIZED',
        'scope': 'offline derivation from finalized source artifacts; no new fault inference',
        'models': models,
        'method': {
            'strict_ranking': 'ascending lexicographic (bit_index, abs(raw_parameter_score)); identical pairs share a code; Spearman uses average ranks',
            'raw_diagnostic': 'abs(raw_parameter_score); ignores bit-significance tiers; not a protection-policy ranking',
            'ties': 'natural score ties retained; stable-identity allocation tie-break excluded',
            'uncertainty': 'original within-recorded-stratum candidate bootstrap; conditional on saved empirical cohort and selected score realization',
            'limitations': ['not population weighted', 'no whole-network protection guarantee',
                           'no inference from overlap or separation of marginal correlation intervals',
                           'candidate bootstrap excludes calibration/eigensolver replication uncertainty',
                           'aggregate asset-lock identity is not silently equated across campaigns'],
        },
        'implementation_files': implementation,
        'runtime': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__},
        'validation': ['deterministic tier-precedence and natural-tie fixtures',
                       'average-tie Spearman checked against scipy.stats.spearmanr',
                       'original MSB point and both original 95% endpoints reproduced within 1e-12 for all models'],
    }
    return {'content': content, 'finalization': {
        'content_sha256': sha(canonical(content)),
        'hash_encoding': 'UTF-8 JSON, sort_keys=True, separators=(comma,colon), allow_nan=False',
        'write_once': True,
    }}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='recompute and compare existing artifact without writes')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print('rank/tie self-tests passed')
        return
    artifact = derive()
    if args.check:
        require(json.loads(OUTPUT.read_text()) == artifact, 'existing derived artifact differs')
        print('deterministic replay verified', artifact['finalization']['content_sha256'])
    else:
        with OUTPUT.open('x') as stream:
            json.dump(artifact, stream, indent=2, allow_nan=False)
            stream.write('\n')
        print('finalized', OUTPUT, artifact['finalization']['content_sha256'])


if __name__ == '__main__':
    main()
