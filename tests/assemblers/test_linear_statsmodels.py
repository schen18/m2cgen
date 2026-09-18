import numpy as np
import pytest
import statsmodels.api as sm
from statsmodels.regression.process_regression import ProcessMLE

from m2cgen import assemblers, ast

from tests import utils


def test_wo_const():
    estimator = utils.StatsmodelsSklearnLikeWrapper(sm.GLS, {})
    fitted = utils.assert_model_predictions_match(
        utils.get_regression_model_trainer(), estimator)

    assembler = assemblers.StatsmodelsModelAssemblerSelector(fitted)
    assembler.assemble()


def test_w_const():
    estimator = utils.StatsmodelsSklearnLikeWrapper(
        sm.GLS,
        dict(init=dict(fit_intercept=True)))
    fitted = utils.assert_model_predictions_match(
        utils.get_regression_model_trainer(), estimator)

    assembler = assemblers.StatsmodelsModelAssemblerSelector(fitted)
    assembler.assemble()


def test_unknown_constant_position():
    estimator = utils.StatsmodelsSklearnLikeWrapper(
        sm.GLS,
        dict(init=dict(hasconst=True)))
    _, __, estimator = utils.get_regression_model_trainer()(estimator)

    with pytest.raises(ValueError, match="Unknown constant position"):
        assemblers.StatsmodelsModelAssemblerSelector(estimator)


def test_unknown_model():

    class ValidGLS(sm.GLS):
        pass

    estimator = utils.StatsmodelsSklearnLikeWrapper(ValidGLS, {})
    _, __, estimator = utils.get_regression_model_trainer()(estimator)

    with pytest.raises(NotImplementedError, match="Model 'ValidGLS' is not supported"):
        assemblers.StatsmodelsModelAssemblerSelector(estimator)


def test_processmle():
    estimator = utils.StatsmodelsSklearnLikeWrapper(
        ProcessMLE,
        dict(init=dict(exog_scale=np.ones(
            (len(utils.get_regression_model_trainer().y_train), 2)),
                       exog_smooth=np.ones(
            (len(utils.get_regression_model_trainer().y_train), 2)),
                       exog_noise=np.ones(
            (len(utils.get_regression_model_trainer().y_train), 2)),
                       time=np.arange(
            len(utils.get_regression_model_trainer().y_train)) % 3,
                       groups=np.arange(
            len(utils.get_regression_model_trainer().y_train)) // 3),
             fit=dict(maxiter=2)))
    fitted = utils.assert_model_predictions_match(
        utils.get_regression_model_trainer(), estimator)

    assemblers.ProcessMLEModelAssembler(fitted).assemble()


def test_glm_logit_link_func():
    estimator = utils.StatsmodelsSklearnLikeWrapper(
        sm.GLM,
        dict(init=dict(
            family=sm.families.Binomial(
                sm.families.links.logit())),
             fit=dict(maxiter=1)))
    estimator = estimator.fit([[1], [2]], [0.1, 0.2])

    assembler = assemblers.StatsmodelsModelAssemblerSelector(estimator)
    actual = assembler.assemble()

    expected = ast.SigmoidExpr(
        ast.BinNumExpr(
            ast.NumVal(0.0),
            ast.BinNumExpr(
                ast.FeatureRef(0),
                ast.NumVal(-0.8567815987),
                ast.BinNumOpType.MUL),
            ast.BinNumOpType.ADD))

    assert utils.cmp_exprs(actual, expected)


def test_glm_power_link_func():
    estimator = utils.StatsmodelsSklearnLikeWrapper(
        sm.GLM,
        dict(init=dict(
            family=sm.families.Tweedie(
                sm.families.links.Power(3))),
             fit=dict(maxiter=1)))
    estimator = estimator.fit([[1], [2]], [0.1, 0.2])

    assembler = assemblers.StatsmodelsModelAssemblerSelector(estimator)
    actual = assembler.assemble()

    expected = ast.PowExpr(
        ast.BinNumExpr(
            ast.NumVal(0.0),
            ast.BinNumExpr(
                ast.FeatureRef(0),
                ast.NumVal(0.0020808009),
                ast.BinNumOpType.MUL),
            ast.BinNumOpType.ADD),
        ast.NumVal(0.3333333333))

    assert utils.cmp_exprs(actual, expected)


def test_glm_negative_power_link_func():
    estimator = utils.StatsmodelsSklearnLikeWrapper(
        sm.GLM,
        dict(init=dict(
            family=sm.families.Tweedie(
                sm.families.links.Power(-3))),
             fit=dict(maxiter=1)))
    estimator = estimator.fit([[1], [2]], [0.1, 0.2])

    assembler = assemblers.StatsmodelsModelAssemblerSelector(estimator)
    actual = assembler.assemble()

    expected = ast.BinNumExpr(
        ast.NumVal(1.0),
        ast.PowExpr(
            ast.BinNumExpr(
                ast.NumVal(0.0),
                ast.BinNumExpr(
                    ast.FeatureRef(0),
                    ast.NumVal(71.0542398846),
                    ast.BinNumOpType.MUL),
                ast.BinNumOpType.ADD),
            ast.NumVal(0.3333333333)),
        ast.BinNumOpType.DIV)

    assert utils.cmp_exprs(actual, expected)


def test_glm_inverse_power_link_func():
    estimator = utils.StatsmodelsSklearnLikeWrapper(
        sm.GLM,
        dict(init=dict(
            family=sm.families.Tweedie(
                sm.families.links.Power(-1))),
             fit=dict(maxiter=1)))
    estimator = estimator.fit([[1], [2]], [0.1, 0.2])

    assembler = assemblers.StatsmodelsModelAssemblerSelector(estimator)
    actual = assembler.assemble()

    expected = ast.BinNumExpr(
        ast.NumVal(1.0),
        ast.BinNumExpr(
            ast.NumVal(0.0),
            ast.BinNumExpr(
                ast.FeatureRef(0),
                ast.NumVal(3.0460921844),
                ast.BinNumOpType.MUL),
            ast.BinNumOpType.ADD),
        ast.BinNumOpType.DIV)

    assert utils.cmp_exprs(actual, expected)


def test_glm_inverse_squared_link_func():
    estimator = utils.StatsmodelsSklearnLikeWrapper(
        sm.GLM,
        dict(init=dict(
            family=sm.families.Tweedie(
                sm.families.links.Power(-2))),
             fit=dict(maxiter=1)))
    estimator = estimator.fit([[1], [2]], [0.1, 0.2])

    assembler = assemblers.StatsmodelsModelAssemblerSelector(estimator)
    actual = assembler.assemble()

    expected = ast.BinNumExpr(
        ast.NumVal(1.0),
        ast.SqrtExpr(
            ast.BinNumExpr(
                ast.NumVal(0.0),
                ast.BinNumExpr(
                    ast.FeatureRef(0),
                    ast.NumVal(15.1237331741),
                    ast.BinNumOpType.MUL),
                ast.BinNumOpType.ADD)),
        ast.BinNumOpType.DIV)

    assert utils.cmp_exprs(actual, expected)


def test_glm_sqr_power_link_func():
    estimator = utils.StatsmodelsSklearnLikeWrapper(
        sm.GLM,
        dict(init=dict(
            family=sm.families.Tweedie(
                sm.families.links.Power(2))),
             fit=dict(maxiter=1)))
    estimator = estimator.fit([[1], [2]], [0.1, 0.2])

    assembler = assemblers.StatsmodelsModelAssemblerSelector(estimator)
    actual = assembler.assemble()

    expected = ast.SqrtExpr(
        ast.BinNumExpr(
            ast.NumVal(0.0),
            ast.BinNumExpr(
                ast.FeatureRef(0),
                ast.NumVal(0.0154915480),
                ast.BinNumOpType.MUL),
            ast.BinNumOpType.ADD))

    assert utils.cmp_exprs(actual, expected)


def test_glm_identity_link_func():
    estimator = utils.StatsmodelsSklearnLikeWrapper(
        sm.GLM,
        dict(init=dict(
            family=sm.families.Tweedie(
                sm.families.links.Power(1))),
             fit=dict(maxiter=1)))
    estimator = estimator.fit([[1], [2], [3]], [0.1, 0.2, 0.2])

    assembler = assemblers.StatsmodelsModelAssemblerSelector(estimator)
    actual = assembler.assemble()

    expected = ast.BinNumExpr(
        ast.NumVal(0.0),
        ast.BinNumExpr(
            ast.FeatureRef(0),
            ast.NumVal(0.0791304348),
            ast.BinNumOpType.MUL),
        ast.BinNumOpType.ADD)

    assert utils.cmp_exprs(actual, expected)


def test_glm_sqrt_link_func():
    estimator = utils.StatsmodelsSklearnLikeWrapper(
        sm.GLM,
        dict(init=dict(
            family=sm.families.Poisson(
                sm.families.links.sqrt())),
             fit=dict(maxiter=1)))
    estimator = estimator.fit([[1], [2]], [0.1, 0.2])

    assembler = assemblers.StatsmodelsModelAssemblerSelector(estimator)
    actual = assembler.assemble()

    expected = ast.PowExpr(
        ast.BinNumExpr(
            ast.NumVal(0.0),
            ast.BinNumExpr(
                ast.FeatureRef(0),
                ast.NumVal(0.2429239017),
                ast.BinNumOpType.MUL),
            ast.BinNumOpType.ADD),
        ast.NumVal(2))

    assert utils.cmp_exprs(actual, expected)


def test_glm_log_link_func():
    estimator = utils.StatsmodelsSklearnLikeWrapper(
        sm.GLM,
        dict(init=dict(
            family=sm.families.Poisson(
                sm.families.links.log())),
             fit=dict(maxiter=1)))
    estimator = estimator.fit([[1], [2]], [0.1, 0.2])

    assembler = assemblers.StatsmodelsModelAssemblerSelector(estimator)
    actual = assembler.assemble()

    expected = ast.ExpExpr(
        ast.BinNumExpr(
            ast.NumVal(0.0),
            ast.BinNumExpr(
                ast.FeatureRef(0),
                ast.NumVal(-1.0242053933),
                ast.BinNumOpType.MUL),
            ast.BinNumOpType.ADD))

    assert utils.cmp_exprs(actual, expected)


def test_glm_cloglog_link_func():
    estimator = utils.StatsmodelsSklearnLikeWrapper(
        sm.GLM,
        dict(init=dict(
            family=sm.families.Binomial(
                sm.families.links.cloglog())),
             fit=dict(maxiter=1)))
    estimator = estimator.fit([[1], [2]], [0.1, 0.2])

    assembler = assemblers.StatsmodelsModelAssemblerSelector(estimator)
    actual = assembler.assemble()

    expected = ast.BinNumExpr(
        ast.NumVal(1.0),
        ast.ExpExpr(
            ast.BinNumExpr(
                ast.NumVal(0.0),
                ast.ExpExpr(
                    ast.BinNumExpr(
                        ast.NumVal(0.0),
                        ast.BinNumExpr(
                            ast.FeatureRef(0),
                            ast.NumVal(-0.8914468745),
                            ast.BinNumOpType.MUL),
                        ast.BinNumOpType.ADD)),
                ast.BinNumOpType.SUB)),
        ast.BinNumOpType.SUB)

    assert utils.cmp_exprs(actual, expected)


def test_glm_negativebinomial_link_func():
    estimator = utils.StatsmodelsSklearnLikeWrapper(
        sm.GLM,
        dict(init=dict(
            family=sm.families.NegativeBinomial(
                sm.families.links.nbinom())),
             fit=dict(maxiter=1)))
    estimator = estimator.fit([[1], [2]], [0.1, 0.2])

    assembler = assemblers.StatsmodelsModelAssemblerSelector(estimator)
    actual = assembler.assemble()

    expected = ast.BinNumExpr(
        ast.NumVal(-1.0),
        ast.BinNumExpr(
            ast.NumVal(1.0),
            ast.ExpExpr(
                ast.BinNumExpr(
                    ast.NumVal(0.0),
                    ast.BinNumExpr(
                        ast.NumVal(0.0),
                        ast.BinNumExpr(
                            ast.FeatureRef(0),
                            ast.NumVal(-1.1079583217),
                            ast.BinNumOpType.MUL),
                        ast.BinNumOpType.ADD),
                    ast.BinNumOpType.SUB)),
            ast.BinNumOpType.SUB),
        ast.BinNumOpType.DIV)

    assert utils.cmp_exprs(actual, expected)


def test_glm_cauchy_link_func():
    estimator = utils.StatsmodelsSklearnLikeWrapper(
        sm.GLM,
        dict(init=dict(
            family=sm.families.Binomial(
                sm.families.links.cauchy())),
             fit=dict(maxiter=1)))
    estimator = estimator.fit([[1], [2]], [0.1, 0.2])

    assembler = assemblers.StatsmodelsModelAssemblerSelector(estimator)
    actual = assembler.assemble()

    expected = ast.BinNumExpr(
        ast.NumVal(0.5),
        ast.BinNumExpr(
            ast.AtanExpr(
                ast.BinNumExpr(
                    ast.NumVal(0.0),
                    ast.BinNumExpr(
                        ast.FeatureRef(0),
                        ast.NumVal(-0.7279996905393095),
                        ast.BinNumOpType.MUL),
                    ast.BinNumOpType.ADD)),
            ast.NumVal(3.141592653589793),
            ast.BinNumOpType.DIV),
        ast.BinNumOpType.ADD)

    assert utils.cmp_exprs(actual, expected)


def test_glm_unknown_link_func():

    class ValidPowerLink(sm.families.links.Power):
        pass

    estimator = utils.StatsmodelsSklearnLikeWrapper(
        sm.GLM,
        dict(init=dict(
            family=sm.families.Tweedie(ValidPowerLink(2))),
             fit=dict(maxiter=1)))
    estimator = estimator.fit([[1], [2]], [0.1, 0.2])

    assembler = assemblers.StatsmodelsModelAssemblerSelector(estimator)

    with pytest.raises(ValueError, match="Unsupported link function 'ValidPowerLink'"):
        assembler.assemble()
