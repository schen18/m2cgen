from sklearn.cluster import BisectingKMeans, KMeans, MiniBatchKMeans
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture

from tests import utils

RANDOM_STATE = 1


def test_kmeans():
    estimator = KMeans(n_clusters=3, random_state=RANDOM_STATE)
    utils.assert_model_predictions_match(
        utils.get_regression_model_trainer(), estimator)


def test_minibatch_kmeans():
    estimator = MiniBatchKMeans(n_clusters=3, random_state=RANDOM_STATE)
    utils.assert_model_predictions_match(
        utils.get_regression_model_trainer(), estimator)


def test_bisecting_kmeans():
    estimator = BisectingKMeans(n_clusters=3, random_state=RANDOM_STATE)
    utils.assert_model_predictions_match(
        utils.get_regression_model_trainer(), estimator)


def test_gaussian_mixture_full():
    estimator = GaussianMixture(
        n_components=3, covariance_type="full", random_state=RANDOM_STATE)
    utils.assert_model_predictions_match(
        utils.get_regression_model_trainer(), estimator)


def test_gaussian_mixture_tied():
    estimator = GaussianMixture(
        n_components=3, covariance_type="tied", random_state=RANDOM_STATE)
    utils.assert_model_predictions_match(
        utils.get_regression_model_trainer(), estimator)


def test_gaussian_mixture_diag():
    estimator = GaussianMixture(
        n_components=3, covariance_type="diag", random_state=RANDOM_STATE)
    utils.assert_model_predictions_match(
        utils.get_regression_model_trainer(), estimator)


def test_gaussian_mixture_spherical():
    estimator = GaussianMixture(
        n_components=3, covariance_type="spherical", random_state=RANDOM_STATE)
    utils.assert_model_predictions_match(
        utils.get_regression_model_trainer(), estimator)


def test_bayesian_gaussian_mixture():
    estimator = BayesianGaussianMixture(
        n_components=3, random_state=RANDOM_STATE)
    utils.assert_model_predictions_match(
        utils.get_regression_model_trainer(), estimator)
