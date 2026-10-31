from pathlib import Path
import numpy as np
import pytest

from netsurf.fkeras_baseline import (
    leading_hessian_eigenpairs, principal_weight_hessian_score,
    repository_weight_hessian_score, fkeras_msb_tier_order,
)
from netsurf.identities import BitIdentity
from netsurf.study_config import load_study, StudyConfigurationError


def test_full_dense_network_hessian_and_finite_difference():
    # Analytic Hessian of a tanh network with squared-error loss.
    x = np.array([[1., -2., .3], [-.2, .7, 1.1], [2., .2, -.8]])
    y = np.array([.2, -.4, .8])
    theta = np.array([.1, -.3, .4])
    def gradient(t):
        a = np.tanh(x @ t)
        return x.T @ ((a - y) * (1 - a*a)) / len(y)
    a = np.tanh(x @ theta)
    h = x.T @ np.diag((1-a*a)**2 - 2*(a-y)*a*(1-a*a)) @ x / len(y)
    v = np.array([.2, .8, -.5])
    np.testing.assert_allclose(h @ v, (gradient(theta+1e-5*v)-gradient(theta-1e-5*v))/2e-5, atol=1e-9)
    vectors, d = leading_hessian_eigenpairs(lambda z: h@z, 3, count=3, seed=97)
    np.testing.assert_allclose(principal_weight_hessian_score(theta, vectors, d.eigenvalues), h@theta, atol=1e-12)
    assert max(d.normalized_residuals) < 1e-12
    assert d.orthogonality_error < 1e-12


def test_signed_largest_magnitude_and_lanczos_reproducibility():
    h = np.diag(np.r_[-12., 8., np.linspace(-1, 1, 68)])
    runs = [leading_hessian_eigenpairs(lambda z:h@z, 70, count=2, seed=17) for _ in range(2)]
    np.testing.assert_array_equal(runs[0][0], runs[1][0])
    np.testing.assert_allclose(runs[0][1].eigenvalues, [-12, 8], atol=1e-9)
    assert runs[0][1].hvp_calls > 2
    with pytest.raises(ValueError, match="budget exhausted"):
        leading_hessian_eigenpairs(lambda z:h@z, 70, count=2, seed=17, max_products=3)
    with pytest.raises(ValueError, match="symmetry"):
        leading_hessian_eigenpairs(lambda z: np.array([[1, 4],[0,2]])@z, 2, count=1, seed=17)


def test_zero_and_degenerate_hessian():
    for matrix in (np.zeros((4,4)), np.diag([2.,2.,-2.,0.])):
        v,d = leading_hessian_eigenpairs(lambda z:matrix@z,4,count=4,seed=1)
        np.testing.assert_allclose(v@v.T, np.eye(4), atol=1e-14)
        np.testing.assert_allclose(principal_weight_hessian_score(np.ones(4),v,d.eigenvalues),matrix@np.ones(4))


def test_repository_formula_absolute_before_sum_and_sign_invariance():
    v = np.array([[1,1],[-1,1]])/np.sqrt(2)
    theta, values = np.array([1.,2.]), np.array([1.,-3.])
    paper = principal_weight_hessian_score(theta,v,values)
    repo = repository_weight_hessian_score(theta,v,values)
    reference = sum(abs(np.dot(l*e,theta)*(l*e)) for e,l in zip(v,values))
    np.testing.assert_allclose(repo,reference)
    assert not np.allclose(repo, abs(principal_weight_hessian_score(theta,v,values**2)))
    for fn in (principal_weight_hessian_score, repository_weight_hessian_score):
        np.testing.assert_allclose(fn(theta,v,values),fn(theta,-v,values))
    # Two independent differences can change rankings, not just normalization.
    paper = abs(principal_weight_hessian_score([1.,3.],np.eye(2),[4.,2.]))
    repo = repository_weight_hessian_score([1.,3.],np.eye(2),[4.,2.])
    assert np.argmax(paper) != np.argmax(repo)


def test_exact_full_tiers_and_partial_tier_masks():
    bits=tuple(BitIdentity("kernel",w,b) for w in range(16) for b in range(8))
    baseline=sorted(bits,key=lambda b:(-b.bit_index,b))
    ordered=fkeras_msb_tier_order(bits,{("kernel",w):w+1 for w in range(16)})
    for coverage in (.125,.25):
        n=int(len(bits)*coverage)
        assert set(baseline[:n]) == set(ordered[:n])
    for coverage in (.0625,.1875):
        n=int(len(bits)*coverage)
        assert set(baseline[:n]) != set(ordered[:n])
    assert len(set(ordered)) == len(bits)
    assert [b.bit_index for b in ordered] == sorted([b.bit_index for b in bits], reverse=True)


def test_m97_schema_and_historical_resolution(tmp_path):
    root=Path(__file__).resolve().parents[1]
    source=root/"config/examples/m97_kws_fkeras_protection_smoke.toml"
    config=load_study(source,check_inputs=False)
    assert config.protection.coverages == (.0625,.125,.1875,.25)
    assert config.estimators[-1].hessian["eigenpairs"] == 2
    bad=tmp_path/"bad.toml"
    bad.write_text(source.read_text().replace('"3.5.0"','"3.4.0"',1))
    with pytest.raises(StudyConfigurationError):
        load_study(bad,check_inputs=False)
    bad.write_text(source.read_text().replace('"bit_significance_desc", ', ""))
    with pytest.raises(StudyConfigurationError,match="strict"):
        load_study(bad,check_inputs=False)
    old=load_study(root/"config/experiments/m9_resnet8_static_estimator_validation_v1.toml",check_inputs=False)
    assert all("hessian" not in row for row in old.resolved_document()["estimators"])
