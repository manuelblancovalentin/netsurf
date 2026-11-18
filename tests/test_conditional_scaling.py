import itertools
import numpy as np
import pytest

from netsurf.conditional_scaling import (measure_field, exact_mask_risks, additive_risks,
    compare_oracle, choose, exposure, error_bound, finite_sample_bound, paired_summary, crossed_bootstrap)
from netsurf.restricted_subcube import integrate_response_table, conditional_marginal_from_table
from netsurf.scaling_study import sample_candidates, exposure_gate, _seal_directory, verify_directory
from netsurf.identities import BitIdentity


@pytest.mark.parametrize("k",[6,8,12])
def test_oracle_all_masks_additive(k):
    table=measure_field(lambda s: .4+sum((i-3)*.001 for i in s),k,exhaustive=True)
    masks=tuple(itertools.combinations(range(k),2))
    for p in (0.,.001,.01,.5,1.):
        np.testing.assert_allclose(exact_mask_risks(table,masks,k,p),additive_risks(table,masks,k,p),atol=1e-14)
        for mask in (masks[0],masks[-1]):
            assert exact_mask_risks(table,[mask],k,p)[0]==pytest.approx(integrate_response_table(table,k,p,protected=mask),abs=1e-14)


def test_bound_exhaustive_boolean_truth_tables_and_sharpness():
    k=3
    states=[tuple(i for i in range(k) if a&(1<<i)) for a in range(2**k)]
    for bits in itertools.product((0.,1.),repeat=8):
        table=dict(zip(states,bits))
        for p in (0.,.01,.5,1.):
            for q in ((),(0,),(0,1),(0,1,2)):
                exact=exact_mask_risks(table,[q],k,p)[0]
                approximate=additive_risks(table,[q],k,p)[0]
                assert abs(exact-approximate)<=error_bound(k-len(q),p,1)+1e-14
    sharp=measure_field(lambda s:float(len(s)!=1),6,exhaustive=True)
    for p in (.001,.1,.9,1):
        err=exact_mask_risks(sharp,[()],6,p)[0]-additive_risks(sharp,[()],6,p)[0]
        assert err==pytest.approx(error_bound(6,p,1),abs=1e-14)


@pytest.mark.parametrize("order",[2,3,5])
def test_pure_interactions_are_not_single_candidate_marginals(order):
    table=measure_field(lambda s:float(set(range(order))<=set(s)),6,exhaustive=True)
    assert all(table[(i,)]==table[()]==0 for i in range(6))
    assert exact_mask_risks(table,[()],6,.1)[0]==pytest.approx(.1**order)
    assert conditional_marginal_from_table(table,6,.1,0)==pytest.approx(.1**(order-1))
    assert exact_mask_risks(table,[(0,)],6,.1)[0]==0


def test_large_pool_linear_cost_negative_values_and_failure_regime():
    calls=[]
    table=measure_field(lambda s:(calls.append(s) or (1-len(s)/128)),64)
    assert len(calls)==65 and table[(0,)]-table[()]<0
    assert choose([-3,-1,-2],2)==(1,2)
    # Pure order-3 damage at bits 3,4,5; stable all-zero additive scores pick 0,1.
    table=measure_field(lambda s:float({3,4,5}<=set(s)),6,exhaustive=True)
    audit=compare_oracle([table],{s:v for s,v in table.items() if len(s)<=1},6,2,.5,(0,1))
    assert audit["diagnostics"]["outside"]["finite_field_selection_regret"]==pytest.approx(.125)
    with pytest.raises(ValueError):measure_field(lambda s:0,64,exhaustive=True)


def test_large_pool_analytic_failure_and_same_mask_boundary():
    # No 2**64 allocation: pair interaction has a closed-form Bernoulli risk.
    table=measure_field(lambda s:float({42,43}<=set(s)),64)
    mask=choose([table[(i,)]-table[()] for i in range(64)],21)
    assert 42 not in mask and 43 not in mask
    p=.1
    actual_regret=p*p  # protecting bit42 would eliminate this pure pair risk
    assert actual_regret>0 and actual_regret<=2*error_bound(43,p,1)
    assert paired_summary([[.3,.3],[.4,.4]],["same_a","same_b"])[0]["mean_difference"]==0


def test_exposure_and_regret_bound():
    assert exposure(176128,22016,expected=100)["logical_ber"]==100/154112
    assert exposure(618880,77360,probability=.001)["expected_applied_flips"]==541.52
    assert error_bound(0,1,1)==error_bound(1,1,1)==0
    with pytest.raises(ValueError):exposure(10,10,expected=1)
    b=finite_sample_bound(6,2,.001,1,32)
    assert b["restricted_regret_bound"]>=2*b["approximation_error_bound"]


def test_sampling_fixed_budget_reproducible_and_positive_support():
    universe=tuple(BitIdentity(f"t{i//8}",i//8,i%8) for i in range(128))
    a=sample_candidates(universe,set(universe[:32]),64,7)
    assert a==sample_candidates(universe,set(universe[:32]),64,7)
    assert len(a["original_mask"])==21 and len({r["identity"] for r in a["candidates"]})==64
    assert all(r["sampling_weight"]*r["inclusion_probability"]==1 for r in a["candidates"])
    assert a!=sample_candidates(universe,set(universe[:32]),64,8)


def test_exposure_and_immutable_checksums(tmp_path):
    ledger={"exposed_test_indices":[1,2]}
    exposure_gate({"selection":[1]},ledger,"engineering")
    with pytest.raises(ValueError):exposure_gate({"final":[3]},ledger,"engineering")
    with pytest.raises(ValueError):exposure_gate({"final":[1]},ledger,"confirmation")
    payload={"schema_version":"1.0.0","protocol":"outside-conditioned-forced-clean/v1","checkpoint_sha256":"test"}
    _seal_directory(tmp_path/"bundle",payload)
    assert verify_directory(tmp_path/"bundle")==payload
    with pytest.raises(FileExistsError):_seal_directory(tmp_path/"bundle",payload)
    (tmp_path/"bundle/scientific_payload.json").chmod(0o600)  # Deliberately corrupt only this disposable fixture.
    (tmp_path/"bundle/scientific_payload.json").write_text("{}")
    with pytest.raises(ValueError):verify_directory(tmp_path/"bundle")


def test_paired_field_and_complete_file_resampling():
    rows=paired_summary([[0,1],[1,0],[1,.5]],["a","b"])
    assert rows[0]["paired_fields"]==3
    assert rows[0]["mean_difference"]==pytest.approx(1/6)
    assert paired_summary([[.2,.2]],["a","b"])[0]["simultaneous_interval"] is None
    scores=np.array([[[0,1,2,3],[0,1,2,3]],[[0,1,2,3],[0,1,2,3]]],dtype=float)
    assert np.all(crossed_bootstrap(scores,labels=[0,0,1,1],seed=7,replicates=10)==0)
    assert np.array_equal(crossed_bootstrap(scores,seed=8,replicates=10),crossed_bootstrap(scores,seed=8,replicates=10))
