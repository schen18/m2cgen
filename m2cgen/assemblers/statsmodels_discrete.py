import numpy as np

from m2cgen import ast
from m2cgen.assemblers import utils
from m2cgen.assemblers.base import ModelAssembler
from m2cgen.assemblers.linear import _linear_to_ast


class BaseStatsmodelsDiscreteModelAssembler(ModelAssembler):
    """Common behavior of the statsmodels discrete-choice assemblers:
    the intercept is taken from the constant column of the design matrix,
    like the other statsmodels assemblers."""

    def __init__(self, model):
        super().__init__(model)
        const_idx = self.model.model.data.const_idx
        if const_idx is None and self.model.k_constant:
            raise ValueError("Unknown constant position")
        self.const_idx = const_idx

    def _coef_and_intercept(self, params):
        params = np.asarray(params, dtype=np.float64)
        idxs = np.arange(len(params))
        intercept = (params[self.const_idx]
                     if self.model.k_constant and self.const_idx is not None
                     else 0.0)
        coef = (params[idxs != self.const_idx]
                if self.model.k_constant and self.const_idx is not None
                else params)
        return coef, intercept


class StatsmodelsBinaryModelAssembler(BaseStatsmodelsDiscreteModelAssembler):
    """Logit: probability of the positive class (``predict``), i.e. the
    sigmoid of the linear predictor."""

    def assemble(self):
        if type(self.model.model).__name__ != "Logit":
            raise NotImplementedError(
                f"Discrete model '{type(self.model.model).__name__}' is not "
                f"supported (Probit requires the erf function which is not "
                f"available in all target languages)")
        coef, intercept = self._coef_and_intercept(self.model.params)
        return ast.SigmoidExpr(_linear_to_ast(coef, intercept))


class StatsmodelsMultinomialModelAssembler(BaseStatsmodelsDiscreteModelAssembler):
    """MNLogit: class probabilities (``predict``) as a softmax over the
    per-class linear predictors with a zero base class."""

    def assemble(self):
        params = np.asarray(self.model.params, dtype=np.float64)   # (k_exog, K-1)
        n_classes = params.shape[1] + 1

        exprs = []
        for k in range(1, n_classes):
            coef, intercept = self._coef_and_intercept(params[:, k - 1])
            exprs.append(_linear_to_ast(coef, intercept))

        base = ast.NumVal(0.0)
        return ast.SoftmaxExpr([base] + exprs)


class StatsmodelsCountModelAssembler(BaseStatsmodelsDiscreteModelAssembler):
    """Poisson / NegativeBinomial: expected count (``predict``) as the
    exponential of the linear predictor. The dispersion parameter of
    NegativeBinomial (the last element of params) is dropped."""

    def assemble(self):
        params = np.asarray(self.model.params, dtype=np.float64)
        if type(self.model.model).__name__ == "NegativeBinomial":
            params = params[:-1]
        coef, intercept = self._coef_and_intercept(params)
        return ast.ExpExpr(_linear_to_ast(coef, intercept))


class StatsmodelsOrderedModelAssembler(BaseStatsmodelsDiscreteModelAssembler):
    """OrderedModel (logit): class probabilities (``predict``) from the
    cumulative probabilities sigmoid(threshold_k - eta)."""

    def __init__(self, model):
        super().__init__(model)
        distr = getattr(self.model.model.distr, "name", None)
        if distr is None:
            distr = type(self.model.model.distr).__name__
        if distr not in {"logit", "logistic"}:
            raise NotImplementedError(
                f"OrderedModel is only supported with the logit "
                f"distribution, got '{distr}' (probit requires the erf "
                f"function which is not available in all target languages)")

    def assemble(self):
        model = self.model.model
        params = np.asarray(self.model.params, dtype=np.float64)
        # params = (exog_coef, transformed thresholds); the thresholds are
        # cumulative in the transformed parameterization
        n_exog = len(params) - (model.k_levels - 1)
        coef, _ = self._coef_and_intercept_exog_only(params[:n_exog])
        thresholds = model.transform_threshold_params(params)[1:-1]

        eta = _linear_to_ast(coef, 0.0)

        # cumulative probabilities P(y <= k) = sigmoid(threshold_k - eta);
        # each is reused for two adjacent class probabilities
        cdf_exprs = [
            ast.SigmoidExpr(
                utils.sub(ast.NumVal(thresholds[k]), eta),
                to_reuse=True)
            for k in range(len(thresholds))
        ]

        exprs = [cdf_exprs[0]]
        for k in range(1, len(cdf_exprs)):
            exprs.append(utils.sub(cdf_exprs[k], cdf_exprs[k - 1]))
        exprs.append(utils.sub(ast.NumVal(1.0), cdf_exprs[-1]))

        return ast.VectorVal(exprs)

    def _coef_and_intercept_exog_only(self, exog_params):
        idxs = np.arange(len(exog_params))
        if self.model.k_constant and self.const_idx is not None:
            intercept = exog_params[self.const_idx]
            coef = exog_params[idxs != self.const_idx]
        else:
            intercept = 0.0
            coef = exog_params
        return coef, intercept
