"""Exact mathematical and offline-only contracts; no TensorFlow required."""
import itertools
import json
import subprocess
import sys

import numpy as np
import pytest

from netsurf.cohort_risk import cohort_risk
from netsurf.interaction_attribution import (
    contrast, decompose_field, mean_interval, mobius_coefficients,
    ordering_comparison, selection_diagnostics,
)
from netsurf.interaction_evidence import file_bootstrap, _csv_records, read_selection
from netsurf.manifests import digest_object
from netsurf.result_bundle import _write_csv, _write_json
from netsurf.restricted_subcube import subcube_states, integrate_response_table


def _clean(d, offset=.3):
    return {():offset, **{(i,):offset+v for i,v in enumerate(d)}}


@pytest.mark.parametrize("p", [0., .001, .1, 1.])
@pytest.mark.parametrize("kind", ["additive", "outside", "pair", "higher", "beneficial"])
def test_decomposition_every_mask_and_pair(p, kind):
    k=4;d=np.array([.2,-.1,.05,.3]);clean=_clean(d)
    def risk(state):
        s=set(state);v=clean[()]+sum(d[i] for i in s)
        if kind=="outside":v+=.7+sum([-.4,.2,.1,-.3][i] for i in s)
        if kind=="pair" and {0,2}<=s:v+=.6
        if kind=="higher" and {0,1,2}<=s:v+=-.8
        if kind=="beneficial" and {1,3}<=s:v-=1.2
        return v
    table={s:risk(s) for s in subcube_states(k)}
    rows=decompose_field(table,clean,k=k,budget=1,probability=p)
    for r in rows:
        expected=integrate_response_table(table,k,p,protected=tuple(r["mask"]))
        assert r["exact_risk"]==pytest.approx(expected,abs=1e-14)
        assert r["clean_additive_risk"]+r["background_intercept_shift"]+r["outside_correction"]+r["within_correction"]==pytest.approx(expected,abs=1e-14)
        if kind in {"additive","outside"}: assert abs(r["within_correction"])<1e-14
        if kind in {"additive","pair","higher","beneficial"}: assert abs(r["outside_correction"])<1e-14
        if kind=="higher": assert r["within_correction"]==pytest.approx(-.8*p**3 if not set(r["mask"]) & {0,1,2} else 0,abs=1e-14)
    for a,b in itertools.product(rows,repeat=2):
        c=contrast(a,b)
        assert c["exact_contrast"]==pytest.approx(c["clean_term"]+c["outside_correction"]+c["within_correction"],abs=1e-14)
        if a is b: assert c["exact_contrast"]==0
    m=mobius_coefficients(table,k)
    for s,v in table.items(): assert sum(c for t,c in m.items() if set(t)<=set(s))==pytest.approx(v,abs=1e-14)


def test_nonadditive_whole_cohort_auc_and_ties():
    labels=[0,0,1,1]
    def scores(s):
        return [0.,2.,1.-(2 if 0 in s else 0),3.-(2 if {1,2}<=set(s) else 0)]
    table={s:cohort_risk(labels,scores(s)) for s in subcube_states(3)}
    clean={s:table[s] for s in [(),(0,),(1,),(2,)]}
    rows=decompose_field(table,clean,k=3,budget=1,probability=.1)
    assert any(abs(r["within_correction"])>0 for r in rows)
    assert cohort_risk([0,1],[1.,1.])==.5
    # AUC's mean across files is not a valid calculation; whole cohort contains
    # the cross-file ordering effects even with a single fixed fault state.
    for r in rows:
        available=set(range(3))-set(r["mask"])
        expected=sum(.1**len(s)*.9**(len(available)-len(s))*v for s,v in table.items() if set(s)<=available)
        assert r["exact_risk"]==pytest.approx(expected)


def test_fail_closed_incomplete_nonfinite_and_duplicate_masks():
    with pytest.raises(ValueError):decompose_field({():0},_clean([1,2]),k=2,budget=1,probability=.1)
    table={s:float(len(s)) for s in subcube_states(2)}
    with pytest.raises(ValueError):decompose_field(table,{():0},k=2,budget=1,probability=.1)
    with pytest.raises(ValueError):decompose_field(table,_clean([1,2]),k=2,budget=1,probability=-.1)
    table[(0,)]=np.nan
    with pytest.raises(ValueError):mobius_coefficients(table,2)


def test_selection_noise_is_not_mask_replication():
    masks=[(0,),(1,),(2,)]
    x=np.array([[0,1,.5],[1,0,.5],[0,1,.5],[1,0,.5]],float)
    a=selection_diagnostics(x,masks,(0,),seed=42,replicates=100)
    b=selection_diagnostics(x,masks,(0,),seed=42,replicates=100)
    assert a==b and a["fields"]==4 and a["masks"]==3
    assert a["near_tied_best_count"]==3
    assert sum(a["bootstrap"]["winner_counts"])==100
    assert a["best_runner_up_gap"]["standard_error"]==pytest.approx(np.std(x[:,1]-x[:,0],ddof=1)/2)
    assert a["cumulative_switches"]>=0
    # An exact all-mask tie is explicitly reported, not evidence of stable benefit.
    z=selection_diagnostics(np.zeros((4,3)),masks,(0,),seed=42,replicates=20)
    assert z["near_tied_best_count"]==3 and z["gap_to_paired_field_se"] is None
    assert z["bootstrap"]["selected_near_best_fraction"]==1


def test_order_reversals_ties_and_n1_uncertainty():
    r=ordering_comparison([0,0,1],[1,1,0])
    assert r["strict_reversals"]==2 and r["prediction_near_tie_pairs"]==1
    assert ordering_comparison([1,1],[0,1])["spearman_raw_tie_ranks"] is None
    assert mean_interval([.1])["confidence_interval"] is None
    assert mean_interval([0,0])["degenerate_empirical_variance"] is True


def test_complete_file_bootstrap_recomputes_paired_auc():
    labels=[0,0,1,1]
    evaluations={"a":{"scores":[0,2,1,3]},"b":{"scores":[0,3,1,2]}}
    name="restricted_selected_vs_original_s5"
    steps=[{"comparison":name,"logical_ber":.1,"removed_schedule_sha256":"a","added_schedule_sha256":"b"}]*2
    result=file_bootstrap(labels,evaluations,steps,2,seed=5,replicates=30)
    rng=np.random.Generator(np.random.PCG64(5));expected=[]
    for _ in range(30):
        n=rng.choice([0,1],2);a=rng.choice([2,3],2);idx=np.r_[n,a]
        expected.append(.1*(cohort_risk(labels,np.array(evaluations["a"]["scores"])[idx])-cohort_risk(labels,np.array(evaluations["b"]["scores"])[idx])))
    np.testing.assert_allclose(np.array(result["contrast_draws"])[:,0],expected,atol=1e-16)
    assert result==file_bootstrap(labels,evaluations,steps,2,seed=5,replicates=30)
    assert np.all(np.array(result["contrast_draws"])[:,1:]==0)


def test_csv_native_type_reconstruction(tmp_path):
    import csv
    row={"schema_version":"6.8.0","phase":"model_selection_isolated","field":"-1","field_seed":"4",
         "replicate":"","local_field":"","candidate_state":"[]","candidate_bits":"['kernel[0]#b1']","risk":"0.5","logical_ber":"0.01","cache_hit":"False"}
    path=tmp_path/"responses.csv"
    with path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=row);writer.writeheader();writer.writerow(row)
    r=_csv_records(path)[0]
    assert "replicate" not in r and "schema_version" not in r
    assert r["cache_hit"] is False and r["risk"]==.5 and r["candidate_state"]==[]
    assert r["candidate_bits"]==['kernel[0]#b1']


def test_side_effect_free_offline_import_and_help():
    code="import sys; import netsurf.interaction_attribution, netsurf.interaction_evidence; assert 'tensorflow' not in sys.modules"
    subprocess.run([sys.executable,"-c",code],check=True)
    result=subprocess.run([sys.executable,"-m","netsurf.interaction_attribution","--help"],capture_output=True,text=True,check=True)
    assert "--campaign" in result.stdout and "--output" in result.stdout


def _reader_fixture(root, corruption=None):
    ids=["tensor/kernel[0]#b0","tensor/kernel[1]#b1"]
    campaign={"design":{"logical_ber":.1,"selection_fields_per_seed":1,"selection_replicates":2},"protected_population":1}
    pool={"candidates":[{"identity":x} for x in ids],"candidate_order_sha256":digest_object(ids),"original_mask_indices":[0]}
    backgrounds=[];responses=[];derived=[]
    for state in [(),(0,),(1,)]:
        responses.append({"phase":"model_selection_isolated","logical_ber":.1,"field":-1,"field_seed":10,
            "candidate_state":list(state),"candidate_bits":[ids[i] for i in state],
            "outside_background_sha256":digest_object([]),"risk":float(sum(state)+len(state)),
            "schedule_sha256":"unused","proposed_events":len(state),"applied_events":len(state),
            "suppressed_events":0,"unique_applied_bits":len(state),"cache_hit":False})
    for f in range(2):
        law={"name":"independent_bernoulli","logical_ber":.1,"candidate_probability":.1}
        bg={"phase":"model_selection","logical_ber":.1,"field":f,"replicate":f,"local_field":0,
            "seed":20+f,"outside_events":[],"outside_event_count":0,"outside_sha256":digest_object([]),
            "sample_sha256":"sample","protection_context_sha256":"context","probability_law":law,"probability_law_sha256":digest_object(law)}
        bg["pairing_key"]=digest_object({k:bg[k] for k in ("phase","logical_ber","field","replicate","seed","outside_sha256","sample_sha256","protection_context_sha256")})
        backgrounds.append(bg)
        table={s:float(len(s)+sum(s)+f) for s in subcube_states(2)}
        for s,risk in table.items():
            responses.append({"phase":"model_selection","logical_ber":.1,"field":f,"field_seed":20+f,"replicate":f,"local_field":0,
                "candidate_state":list(s),"candidate_bits":[ids[i] for i in s],"outside_background_sha256":bg["outside_sha256"],
                "pairing_key":bg["pairing_key"],"probability_law_sha256":bg["probability_law_sha256"],"protection_context_sha256":"context",
                "risk":risk,"schedule_sha256":"unused","proposed_events":len(s),"applied_events":len(s),"suppressed_events":0,"unique_applied_bits":len(s),"cache_hit":False})
        derived.append({"field":f,"pairing_key":bg["pairing_key"],"response_table_sha256":digest_object([{"state":list(s),"risk":table[s]} for s in subcube_states(2)])})
    summary={"logical_ber":.1,"original_mask_indices":[0],"global_protection_cardinality_preserved":True,"global_protection_cardinality":1,
             "isolated_damages":[{"candidate":ids[0],"damage":1.},{"candidate":ids[1],"damage":2.}]}
    payload={"partitions":{"model_selection":{"sample_sha256":"sample"}},"protection_context":{"identity_sha256":"context"},
             "restricted_oracle":{"summaries":[summary],"response_records_sha256":digest_object(responses),"background_records_sha256":digest_object(backgrounds),"derived_records_sha256":digest_object(derived)}}
    if corruption=="missing":responses.pop()
    if corruption=="duplicate":responses.append(responses[-1])
    if corruption=="pairing":responses[-1]["pairing_key"]="wrong"
    if corruption=="nonfinite":responses[-1]["risk"]=float("nan")
    # Binding hash cannot rescue an internally invalid table: corruptions other
    # than hash loss deliberately update this outer digest, testing semantic gates.
    if corruption in {"duplicate","pairing"}:payload["restricted_oracle"]["response_records_sha256"]=digest_object(responses)
    fields=["schema_version",*dict.fromkeys(k for r in responses for k in r)]
    _write_csv(root/"measurements/restricted_oracle_responses.csv",fields,[{"schema_version":"6.8.0",**r} for r in responses])
    for name,value in [("restricted_oracle_backgrounds",backgrounds),("restricted_oracle_derived",derived),("candidates",[{"identity":i} for i in ids])]:
        _write_json(root/f"artifacts/{name}.json",value)
    return payload,pool,campaign


def test_strict_response_reader_all_fields_and_native_hashes(tmp_path):
    payload,pool,campaign=_reader_fixture(tmp_path)
    rows,clean,tables,metadata,summary=read_selection(tmp_path,payload,pool,campaign)
    assert len(rows)==11 and len(tables)==2 and tables[1][()]==1
    assert clean[(1,)]==2 and metadata[1]["seed"]==21


@pytest.mark.parametrize("corruption",["missing","duplicate","pairing","nonfinite"])
def test_strict_response_reader_rejects_corruption(tmp_path,corruption):
    payload,pool,campaign=_reader_fixture(tmp_path,corruption)
    with pytest.raises((ValueError,ArithmeticError)):
        read_selection(tmp_path,payload,pool,campaign)


def test_offline_output_rejects_overwrite(tmp_path):
    from netsurf.interaction_evidence import analyze_campaigns
    with pytest.raises(FileExistsError):
        analyze_campaigns([tmp_path/"campaign.json"],tmp_path)


def test_hash_pinned_source_inventory(tmp_path):
    from netsurf.interaction_attribution import source_paths
    from netsurf.manifests import sha256_file
    campaign=tmp_path/"campaign.json";campaign.write_text('{}')
    inventory=tmp_path/"sources.json"
    document={"schema_version":"1.0.0","description":"test","campaigns":[{"path":str(campaign),"sha256":sha256_file(campaign)}],"exclusions":[]}
    inventory.write_text(json.dumps(document))
    assert source_paths(inventory)==[campaign]
    campaign.write_text('{"changed":true}')
    with pytest.raises(ValueError,match="hash mismatch"):source_paths(inventory)


def test_file_bootstrap_identical_masks_and_missing_classes():
    a=file_bootstrap([0,1],{},[],fields=32,seed=3,replicates=10)
    assert np.count_nonzero(a["contrast_draws"])==0
    with pytest.raises(ValueError):file_bootstrap([0,0],{},[],fields=32,seed=3,replicates=10)


def test_saved_measurement_report_replay_preserves_every_pool(tmp_path):
    from netsurf.conditional_config import PAIRED_SWAP_COMPARISONS
    from netsurf.interaction_reporting import write_report
    masks = [(0,), (1,), (2,)]
    ordering = ordering_comparison([0, 1, 2], [2, 1, 0])
    stability = selection_diagnostics([[0, 1, 2], [1, 0, 2]], masks, (0,), seed=3, replicates=10)
    pools = []
    for number, effect in enumerate((-.1, .3)):
        pools.append({"pool_id":f"pool_{number:03d}", "toycar":None,
            "final_comparisons":{"comparisons":[{"comparison":c, "mean_effect":mean_interval([effect, effect])}
                                                   for c in PAIRED_SWAP_COMPARISONS]},
            "clean_vs_exact_ordering":ordering, "conditioned_vs_exact_ordering":ordering,
            "all_mask_pair_component_magnitudes":{k:{"mean_absolute_field_mean_contrast":abs(effect)}
                for k in ("clean_term", "outside_correction", "within_correction", "exact_contrast")},
            "isolated_selection_regret":.1, "selection_stability":stability})
    payload={"schema_version":"1.0.0", "campaigns":[{"name":"m96_test", "purpose":"engineering",
        "logical_ber":.1, "risk_metric":"categorical_crossentropy", "pools":pools}]}
    source=tmp_path/"scientific_payload.json"
    _write_json(source,payload)
    before=source.read_bytes()
    for name in ("first", "replay"):
        write_report(json.loads(source.read_text()),tmp_path/name,plots=False)
    assert source.read_bytes()==before
    assert (tmp_path/"first/report.md").read_bytes()==(tmp_path/"replay/report.md").read_bytes()
    summary=json.loads((tmp_path/"first/aggregate_summary.json").read_text())
    assert all(r["pool_summary"]["count"]==2 for r in summary["comparisons"])
    assert all(r["pool_summary"]["mean"]==pytest.approx(.1) for r in summary["comparisons"])
    assert "pool_000" in (tmp_path/"first/report.md").read_text()
    assert "pool_001" in (tmp_path/"first/report.md").read_text()


def test_boundary_masks_and_invalid_diagnostic_units():
    table={s:.2+len(s)-(.4 if {0,1}<=set(s) else 0) for s in subcube_states(2)}
    clean={s:table[s] for s in ((),(0,),(1,))}
    for budget in (0,2):
        rows=decompose_field(table,clean,k=2,budget=budget,probability=.1)
        assert len(rows)==1 and contrast(rows[0],rows[0])["exact_contrast"]==0
    assert rows[0]["exact_risk"]==.2
    with pytest.raises(ValueError):selection_diagnostics([[0],[1]],[(0,)],(0,),seed=1)
    with pytest.raises(ValueError):selection_diagnostics([[0,1],[1,0]],[(0,),(0,)],(0,),seed=1)
    with pytest.raises(ValueError):mean_interval([1,2],confidence=1.1)
