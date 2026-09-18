import math

import numpy as np

from m2cgen import ast
from m2cgen.assemblers import utils
from m2cgen.assemblers.base import ModelAssembler


def _digamma(x):
    """Digamma function (psi), accurate to ~1e-14 for x > 0.

    Implemented locally (asymptotic expansion with upward recurrence for
    small arguments) to keep m2cgen free of a scipy dependency; it is only
    needed at assemble time for BayesianGaussianMixture models.
    """
    result = 0.0
    while x < 6.0:
        result -= 1.0 / x
        x += 1.0
    inv = 1.0 / x
    inv2 = inv * inv
    result += math.log(x) - 0.5 * inv - inv2 * (
        1.0 / 12 - inv2 * (1.0 / 120 - inv2 * (1.0 / 252)))
    return result


class KMeansModelAssembler(ModelAssembler):
    """
    K-means family of clustering models.

    The generated code outputs the Euclidean distances to each cluster
    center (consistent with the output of the ``transform`` method); the
    cluster assignment of the ``predict`` method is the index of the
    smallest value of the output.
    """

    def assemble(self):
        return ast.VectorVal([
            self._distance_to_center(center)
            for center in self.model.cluster_centers_
        ])

    def _distance_to_center(self, center):
        squared_terms = [
            ast.PowExpr(
                utils.sub(ast.NumVal(value), ast.FeatureRef(idx)),
                ast.NumVal(2.0))
            for idx, value in enumerate(center)
        ]
        return ast.SqrtExpr(utils.apply_op_to_expressions(
            ast.BinNumOpType.ADD, *squared_terms))


class GaussianMixtureModelAssembler(ModelAssembler):
    """
    Gaussian mixture models.

    The generated code outputs the cluster responsibilities (consistent
    with the output of the ``predict_proba`` method); the cluster
    assignment of the ``predict`` method is the index of the largest value
    of the output.

    The per-cluster log-densities follow the formulas of
    ``sklearn.mixture.gaussian_mixture._estimate_log_gaussian_prob``:
    for "full"/"tied" covariances the triangular solve with the Cholesky
    factor of the precision matrix is emitted as explicit linear forms.
    """

    def assemble(self):
        model = self.model
        means = model.means_
        n_components, n_features = means.shape
        precisions_chol = model.precisions_cholesky_
        covariance_type = model.covariance_type

        if hasattr(model, "weight_concentration_"):
            # BayesianGaussianMixture: the effective weights and the
            # Wishart normalization terms replace log(weights_) and add
            # per-component constants (all x-independent)
            log_weights = self._bayesian_log_weights(model)
            extra_const = self._bayesian_extra_const(model, n_features)
        else:
            log_weights = np.log(model.weights_)
            extra_const = np.zeros(n_components)

        log_prob_exprs = []
        for k in range(n_components):
            const, terms = self._log_gaussian_prob_terms(
                k, means, precisions_chol, covariance_type, n_features)
            log_prob_exprs.append(utils.apply_op_to_expressions(
                ast.BinNumOpType.ADD,
                ast.NumVal(log_weights[k] + const + extra_const[k]),
                *terms))

        return ast.SoftmaxExpr(log_prob_exprs)

    @staticmethod
    def _bayesian_log_weights(model):
        # mirrors BayesianGaussianMixture._estimate_log_weights
        digamma = np.vectorize(_digamma)
        weight_concentration = model.weight_concentration_
        if model.weight_concentration_prior_type == "dirichlet_process":
            # tuple of the two Dirichlet process concentration parameters
            alpha, beta = (np.asarray(w) for w in weight_concentration)
            digamma_sum = digamma(alpha + beta)
            return (
                digamma(alpha) - digamma_sum
                + np.hstack((0, np.cumsum(digamma(beta) - digamma_sum)[:-1])))
        return digamma(weight_concentration) - digamma(
            np.sum(weight_concentration))

    @staticmethod
    def _bayesian_extra_const(model, n_features):
        # mirrors the normalization terms of
        # BayesianGaussianMixture._estimate_log_prob
        digamma = np.vectorize(_digamma)
        degrees_of_freedom = model.degrees_of_freedom_
        log_lambda = n_features * np.log(2.0) + np.sum(
            digamma(0.5 * (degrees_of_freedom
                           - np.arange(n_features)[:, np.newaxis])), 0)
        return (
            -0.5 * n_features * np.log(degrees_of_freedom)
            + 0.5 * (log_lambda - n_features / model.mean_precision_))

    def _log_gaussian_prob_terms(self, k, means, precisions_chol,
                                 covariance_type, n_features):
        # returns (constant, terms) so that log N_k(x) = constant + sum(terms)
        if covariance_type == "diag":
            precision = precisions_chol[k] ** 2
            const = -0.5 * np.log(2 * np.pi / precision).sum()
            terms = [
                utils.mul(
                    ast.NumVal(-0.5 * precision[j]),
                    self._squared_diff(means[k][j], j))
                for j in range(n_features)
            ]
        elif covariance_type == "spherical":
            precision = float(precisions_chol[k]) ** 2
            const = -0.5 * n_features * np.log(2 * np.pi / precision)
            terms = [
                utils.mul(
                    ast.NumVal(-0.5 * precision),
                    self._squared_diff(means[k][j], j))
                for j in range(n_features)
            ]
        else:  # "full" or "tied"
            chol = precisions_chol[k] if covariance_type == "full" \
                else precisions_chol
            const = np.log(np.diag(chol)).sum() \
                - 0.5 * n_features * np.log(2 * np.pi)
            terms = []
            for j in range(n_features):
                # y_j = sum_{i<=j} diff_i * chol[i, j], matching sklearn's
                # y = (X - mu) @ chol (columns of the Cholesky factor)
                linear = utils.apply_op_to_expressions(
                    ast.BinNumOpType.ADD,
                    *[utils.mul(ast.NumVal(chol[i, j]),
                                utils.sub(ast.NumVal(means[k][i]),
                                          ast.FeatureRef(i)))
                      for i in range(j + 1)])
                terms.append(utils.mul(
                    ast.NumVal(-0.5), ast.PowExpr(linear, ast.NumVal(2.0))))
        return float(const), terms

    @staticmethod
    def _squared_diff(mean_value, feature_idx):
        return ast.PowExpr(
            utils.sub(ast.NumVal(mean_value), ast.FeatureRef(feature_idx)),
            ast.NumVal(2.0))
