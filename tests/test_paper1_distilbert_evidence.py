"""No inference: manuscript archive and paired-statistic regression checks."""
from pathlib import Path
import importlib.util
import json
import struct

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT/'publications/tecs_software_review_2026-11-02'
spec = importlib.util.spec_from_file_location('paper_distilbert', PAPER/'build_distilbert_evidence.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_bound_rejects_tampered_field_or_wrong_plan():
    field = {'plan_sha256':'one', 'accuracy':.5}
    field['payload_sha256'] = audit.digest(field)
    audit.bound(field, plan='one')
    with pytest.raises(ValueError, match='another plan'):
        audit.bound(field, plan='two')
    field['accuracy'] = .9
    with pytest.raises(ValueError, match='mismatch'):
        audit.bound(field, plan='one')


def test_paired_field_statistics_keep_negative_zero_and_tail_outcomes():
    values = np.array([-1.]*8 + [0.]*8 + [3.]*16)
    a = audit.paired_statistics(values, 7)
    assert a == audit.paired_statistics(values, 7)
    assert a['matched_pairs'] == 32 and a['mean'] == 1.25
    assert (a['favorable_fields'], a['adverse_fields']) == (16,8)
    assert a['ci_low'] < a['mean'] < a['ci_high']
    b = audit.paired_statistics(-values, 7)
    assert b['mean'] == -a['mean'] and b['p_unadjusted'] == a['p_unadjusted']
    assert b['ci_low'] == -a['ci_high']
    zero = audit.paired_statistics(np.zeros(32), 7)
    assert zero['ci_low'] == zero['ci_high'] == 0 and zero['p_unadjusted'] == 1


def test_holm_retains_full_family_not_displayed_subset():
    family = [{'p_unadjusted':p} for p in [.01,.04,.03,1.]]
    audit.holm(family)
    assert np.allclose([r['p_holm'] for r in family], [.04,.09,.09,1.])


def test_finalized_distilbert_archive_offline_if_present():
    pin = json.loads((ROOT/'config/paper1_distilbert_source.json').read_text())
    if not Path(pin['path']).is_dir():
        pytest.skip('optional historical archive unavailable; synthetic checks still run')
    receipt, raw, means, pairs, accuracy = audit.audit(pin)
    assert receipt['verified_fields'] == 96 and len(raw) == 768
    assert len(means) == 24 and len(pairs) == len(accuracy) == 12
    r = next(x for x in pairs if x['ber'] == 16/1024 and x['budget'] == 128 and x['baseline']=='significance')
    assert r['matched_pairs'] == 32
    assert r['mean'] == pytest.approx(-.45812229111258507)
    assert r['p_holm'] == pytest.approx(.0011998800119988001)
    assert receipt['new_model_inferences'] == receipt['historical_source_writes'] == 0
    wrong = {**pin, 'ready_snapshot_sha256':'0'*64}
    with pytest.raises(ValueError, match='READY'):
        audit.audit(wrong)


def test_main_manuscript_has_one_fkeras_definition_and_fp32_boundary():
    sections = PAPER/'overleaf/sections'
    main = '\n'.join(p.read_text() for p in sections.glob('*.tex'))
    assert 'FKeras repository' not in main and 'eq:fkerasrepo' not in main
    assert 'FKeras paper &' not in main
    assert 'first 32 scalar' in main and 'capped cross-entropy' in main
    assert 'negative}-class logit' in main
    assert 'Calibration' in main and 'confounded' in main
    assert 'not representative transformer layers' in main
    builder = (PAPER/'build_evidence.py').read_text()
    assert "labels=['MSB-first','Global gradient','Absolute gradient','Positive gradient','FKeras']" in builder


def test_frozen_classifier_slice_is_one_logit_if_checkpoint_present():
    plan = json.loads((ROOT/'config/paper1_distilbert_source.json').read_text())
    source = Path(plan['path'])
    # The frozen archive can be copied to another server; local asset absence is optional.
    model = Path('/data/models/models--distilbert--distilbert-base-uncased-finetuned-sst-2-english/snapshots/714eb0fa89d2f80546fda750413ed43d93601a13/model.safetensors')
    config = model.parent/'config.json'
    if not source.is_dir() or not model.is_file() or not config.is_file():
        pytest.skip('optional pinned DistilBERT asset unavailable')
    with model.open('rb') as stream:
        length = struct.unpack('<Q', stream.read(8))[0]
        header = json.loads(stream.read(length))
    assert header['classifier.weight']['shape'] == [2, 768]
    assert header['classifier.weight']['dtype'] == 'F32'
    assert json.loads(config.read_text())['id2label'] == {'0':'NEGATIVE','1':'POSITIVE'}
    # C-order flat indices 0..31 all have output row 0; row 1 begins at 768.
    assert {(flat // 768, flat % 768) for flat in range(32)} == {(0, col) for col in range(32)}


def test_distilbert_table_keeps_all_original_contrasts_in_nats():
    table = (PAPER/'overleaf/tables/distilbert_paired.tex').read_text().splitlines()
    assert len(table) == 12
    assert all(len(line.split('&')) == 6 for line in table)
    assert audit.effect_number(-6.240210282025538e-05) == '-6.24e-05'
    assert audit.effect_number(-.45812229111258507) == '-0.458'
    assert '10^{-3}' not in '\n'.join(table)
    assert sum('significance' in line for line in table) == 6
    assert sum('random' in line for line in table) == 6


def test_prototype_heatmap_uses_paired_measured_cells_only():
    import csv
    path = PAPER/'evidence/fkeras_heatmap_cells.csv'
    if not path.exists():
        pytest.skip('offline derived figures not built')
    rows = list(csv.DictReader(path.open()))
    assert len(rows) == 24
    assert {r['baseline'] for r in rows} == {'MSB-first'}
    assert {r['method'] for r in rows} == {'FKeras'}
    assert all(int(r['matched_pairs']) == 32 for r in rows)
    for r in rows:
        if float(r['coverage']) in (.125,.25):
            assert float(r['improvement']) == float(r['ci_low']) == float(r['ci_high']) == 0
    losses = [r for r in rows if r['metric']=='categorical_crossentropy' and float(r['ber'])==.01 and float(r['coverage'])==.0625]
    assert len(losses)==1 and float(losses[0]['improvement']) > 1
