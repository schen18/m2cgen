import numpy as np
from sklearn import datasets
from sklearn.ensemble import IsolationForest

from m2cgen import export_to_python


def test_isolation_forest_decision_function():
    X, _ = datasets.make_classification(
        n_samples=100, n_features=5, n_informative=2, random_state=42)
    model = IsolationForest(n_estimators=20, random_state=1).fit(X)

    scope = {}
    exec(export_to_python(model), scope)
    score = scope["score"]

    got = np.array([score(list(x)) for x in X[:20]])
    expected = model.decision_function(X[:20])
    assert np.allclose(expected, got, atol=1e-9)


def test_isolation_forest_custom_contamination():
    X, _ = datasets.make_classification(
        n_samples=100, n_features=5, n_informative=2, random_state=42)
    model = IsolationForest(
        n_estimators=20, contamination=0.1, random_state=1).fit(X)

    scope = {}
    exec(export_to_python(model), scope)
    score = scope["score"]

    got = np.array([score(list(x)) for x in X[:20]])
    expected = model.decision_function(X[:20])
    assert np.allclose(expected, got, atol=1e-9)


def test_isolation_forest_max_samples_one():
    X, _ = datasets.make_classification(
        n_samples=50, n_features=5, n_informative=2, random_state=42)
    model = IsolationForest(n_estimators=5, max_samples=1, random_state=1).fit(X)

    scope = {}
    exec(export_to_python(model), scope)
    score = scope["score"]

    got = np.array([score(list(x)) for x in X[:10]])
    expected = model.decision_function(X[:10])
    assert np.allclose(expected, got, atol=1e-9)
