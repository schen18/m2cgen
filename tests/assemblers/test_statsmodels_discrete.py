import numpy as np
import pytest
from sklearn import datasets
from statsmodels.discrete.discrete_model import Logit, MNLogit, NegativeBinomial, Poisson, Probit
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.robust.robust_linear_model import RLM

from m2cgen import export_to_python
from m2cgen.assemblers.statsmodels_discrete import StatsmodelsBinaryModelAssembler, StatsmodelsOrderedModelAssembler


def _assert_predictions_match(model, X, expected):
    scope = {}
    exec(export_to_python(model), scope)
    score = scope["score"]
    got = np.array([np.ravel(score(list(x))) for x in X[:20]]).squeeze()
    assert np.allclose(np.asarray(expected).squeeze()[:20], got, atol=1e-8)


def _data(binary=False, count=False, ordinal=False):
    X, y = datasets.make_classification(
        n_samples=200, n_features=5, n_informative=2,
        n_classes=3, n_clusters_per_class=1, random_state=42)
    if binary:
        y = (y > 0).astype(float)
    elif count:
        rng = np.random.RandomState(0)
        y = rng.poisson(np.exp(X[:, 0] * 0.5 + 0.3))
    elif ordinal:
        y = (y > 0).astype(int) + 1
    Xc = np.hstack([np.ones((len(X), 1)), X])
    return X, y, Xc


def test_logit():
    X, y, Xc = _data(binary=True)
    res = Logit(y, Xc).fit(disp=0)
    _assert_predictions_match(res, X, res.predict(Xc[:20]))


def test_mnlogit():
    X, y, Xc = _data()
    res = MNLogit(y, Xc).fit(disp=0)
    _assert_predictions_match(res, X, res.predict(Xc[:20]))


def test_poisson():
    X, y, Xc = _data(count=True)
    res = Poisson(y, Xc).fit(disp=0)
    _assert_predictions_match(res, X, res.predict(Xc[:20]))


def test_negative_binomial():
    X, y, Xc = _data(count=True)
    res = NegativeBinomial(y, Xc).fit(disp=0, method="nm", maxiter=200)
    _assert_predictions_match(res, X, res.predict(Xc[:20]))


def test_ordered_model_three_levels():
    X, y = datasets.load_iris(return_X_y=True)
    res = OrderedModel(y, X, distr="logit").fit(disp=0)
    _assert_predictions_match(res, X, res.predict(X[:20]))


def test_ordered_model_two_levels():
    X, y, _ = _data(ordinal=True)
    res = OrderedModel(y, X, distr="logit").fit(disp=0)
    _assert_predictions_match(res, X, res.predict(X[:20]))


def test_rlm():
    X, y, Xc = _data()
    res = RLM(y.astype(float), Xc).fit()
    _assert_predictions_match(res, X, res.predict(Xc[:20]))


def test_probit_not_supported():
    X, y, Xc = _data(binary=True)
    res = Probit(y, Xc).fit(disp=0)
    with pytest.raises(NotImplementedError, match="erf"):
        StatsmodelsBinaryModelAssembler(res).assemble()


def test_ordered_probit_not_supported():
    X, y, _ = _data(ordinal=True)
    res = OrderedModel(y, X, distr="probit").fit(disp=0)
    with pytest.raises(NotImplementedError, match="erf"):
        StatsmodelsOrderedModelAssembler(res).assemble()
