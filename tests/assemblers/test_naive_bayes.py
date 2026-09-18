import numpy as np
from sklearn import datasets
from sklearn.naive_bayes import BernoulliNB, ComplementNB, GaussianNB, MultinomialNB

from m2cgen import export_to_python


def _assert_predictions_match(model, X, expected):
    scope = {}
    exec(export_to_python(model), scope)
    score = scope["score"]
    got = np.array([np.ravel(score(list(x))) for x in X[:20]]).squeeze()
    exp = np.asarray(expected)[:20]
    assert np.allclose(exp.squeeze(), got, atol=1e-9), (exp.squeeze()[:3], got[:3])


def test_gaussian_nb():
    X, y = datasets.make_classification(
        n_samples=100, n_features=5, n_informative=2,
        n_classes=3, n_clusters_per_class=1, random_state=42)
    model = GaussianNB().fit(X, y)
    _assert_predictions_match(model, X, model.predict_proba(X))


def test_multinomial_nb():
    X, y = datasets.make_classification(
        n_samples=100, n_features=5, n_informative=2,
        n_classes=3, n_clusters_per_class=1, random_state=42)
    X = np.abs(X) + 0.5  # multinomial expects non-negative features
    model = MultinomialNB().fit(X, y)
    _assert_predictions_match(model, X, model.predict_proba(X))


def test_complement_nb():
    X, y = datasets.make_classification(
        n_samples=100, n_features=5, n_informative=2,
        n_classes=3, n_clusters_per_class=1, random_state=42)
    X = np.abs(X) + 0.5
    model = ComplementNB().fit(X, y)
    _assert_predictions_match(model, X, model.predict_proba(X))


def test_bernoulli_nb():
    X, y = datasets.make_classification(
        n_samples=100, n_features=5, n_informative=2,
        n_classes=3, n_clusters_per_class=1, random_state=42)
    model = BernoulliNB().fit(X, y)
    _assert_predictions_match(model, X, model.predict_proba(X))


def test_bernoulli_nb_no_binarize():
    X, y = datasets.make_classification(
        n_samples=100, n_features=5, n_informative=2,
        n_classes=3, n_clusters_per_class=1, random_state=42)
    X = (X > 0).astype(float)
    model = BernoulliNB(binarize=None).fit(X, y)
    _assert_predictions_match(model, X, model.predict_proba(X))
