import numpy as np

from m2cgen import ast
from m2cgen.assemblers import utils
from m2cgen.assemblers.base import ModelAssembler
from m2cgen.assemblers.linear import _linear_to_ast


class BaseNaiveBayesModelAssembler(ModelAssembler):
    """
    Naive Bayes classifiers of scikit-learn.

    The generated code outputs class probabilities, consistent with the
    ``predict_proba`` method. All Naive Bayes variants compute the joint
    log-likelihood, which is log-linear in the features, followed by a
    normalization; the variants differ only in how the per-class
    coefficients and intercepts are derived from the fitted attributes.
    The class label (``predict``, the argmax) is not emitted.
    """

    def assemble(self):
        coef, intercept = self._get_coef_and_intercept()
        exprs = [
            _linear_to_ast(coef[idx], intercept[idx])
            for idx in range(coef.shape[0])
        ]
        return ast.SoftmaxExpr(exprs)

    def _get_coef_and_intercept(self):
        """Returns per-class (coef, intercept) of shape (n_classes, n_features)
        and (n_classes,)."""
        raise NotImplementedError


class MultinomialNaiveBayesModelAssembler(BaseNaiveBayesModelAssembler):
    """MultinomialNB: log prior + x . feature_log_prob_."""

    def _get_coef_and_intercept(self):
        return self.model.feature_log_prob_, self.model.class_log_prior_


class ComplementNaiveBayesModelAssembler(BaseNaiveBayesModelAssembler):
    """ComplementNB: x . feature_log_prob_ (no class prior, matching the
    library's _joint_log_likelihood)."""

    def _get_coef_and_intercept(self):
        return (self.model.feature_log_prob_,
                np.zeros(len(self.model.classes_)))


class GaussianNaiveBayesModelAssembler(ModelAssembler):
    """GaussianNB: log prior - 0.5 * sum(log(2 pi var) + (x - mean)^2 / var)."""

    def assemble(self):
        model = self.model
        var = model.var_
        theta = model.theta_
        exprs = []
        for k in range(theta.shape[0]):
            quadratic_terms = [
                utils.mul(
                    ast.NumVal(-0.5 / var[k, j]),
                    ast.PowExpr(
                        utils.sub(ast.NumVal(theta[k, j]), ast.FeatureRef(j)),
                        ast.NumVal(2.0)))
                for j in range(theta.shape[1])
            ]
            intercept = (np.log(model.class_prior_[k])
                         - 0.5 * np.log(2 * np.pi * var[k]).sum())
            exprs.append(utils.apply_op_to_expressions(
                ast.BinNumOpType.ADD, ast.NumVal(intercept), *quadratic_terms))
        return ast.SoftmaxExpr(exprs)


class BernoulliNaiveBayesModelAssembler(ModelAssembler):
    """BernoulliNB: per-feature indicators (x > binarize) with log-odds
    coefficients and log(1 - theta) intercepts; when ``binarize`` is None
    the features are used as-is (assumed to be already binary)."""

    def assemble(self):
        model = self.model
        theta = np.exp(model.feature_log_prob_)      # probabilities
        log_odds = np.log(theta) - np.log1p(-theta)
        intercept = model.class_log_prior_ + np.log1p(-theta).sum(axis=1)
        binarize = model.binarize

        exprs = []
        for k in range(theta.shape[0]):
            terms = []
            for j in range(theta.shape[1]):
                if binarize is not None:
                    terms.append(ast.IfExpr(
                        ast.CompExpr(
                            ast.FeatureRef(j),
                            ast.NumVal(binarize),
                            ast.CompOpType.GT),
                        ast.NumVal(log_odds[k, j]),
                        ast.NumVal(0.0)))
                else:
                    terms.append(utils.mul(
                        ast.NumVal(log_odds[k, j]), ast.FeatureRef(j)))
            exprs.append(utils.apply_op_to_expressions(
                ast.BinNumOpType.ADD, ast.NumVal(intercept[k]), *terms))
        return ast.SoftmaxExpr(exprs)
