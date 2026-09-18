import lightgbm as lgb
import numpy as np
import pytest

from m2cgen import ast
from m2cgen.assemblers import LightGBMModelAssembler

from tests import utils


def test_binary_classification():
    estimator = lgb.LGBMClassifier(n_estimators=2, random_state=1, max_depth=1)
    utils.get_binary_classification_model_trainer()(estimator)

    assembler = LightGBMModelAssembler(estimator)
    actual = assembler.assemble()

    sigmoid = ast.SigmoidExpr(
        ast.BinNumExpr(
            ast.IfExpr(
                ast.CompExpr(
                    ast.FeatureRef(20),
                    ast.NumVal(16.795),
                    ast.CompOpType.GT),
                ast.NumVal(0.27502096830384837),
                ast.NumVal(0.6391171126839048)),
            ast.IfExpr(
                ast.CompExpr(
                    ast.FeatureRef(27),
                    ast.NumVal(0.14205),
                    ast.CompOpType.GT),
                ast.NumVal(-0.21340153096570616),
                ast.NumVal(0.11583109256834748)),
            ast.BinNumOpType.ADD),
        to_reuse=True)

    expected = ast.VectorVal([
        ast.BinNumExpr(ast.NumVal(1), sigmoid, ast.BinNumOpType.SUB),
        sigmoid])

    assert utils.cmp_exprs(actual, expected)


def test_multi_class():
    estimator = lgb.LGBMClassifier(n_estimators=1, random_state=1, max_depth=1)
    estimator.fit(np.array([[1], [2], [3]]), np.array([1, 2, 3]))

    assembler = LightGBMModelAssembler(estimator)
    actual = assembler.assemble()

    num_expr = ast.NumVal(-1.0986122886681098)
    expected = ast.SoftmaxExpr([num_expr] * 3)

    assert utils.cmp_exprs(actual, expected)


def test_regression():
    estimator = lgb.LGBMRegressor(n_estimators=2, random_state=1, max_depth=1)
    utils.assert_model_predictions_match(
        utils.get_regression_model_trainer(), estimator)


def test_regression_random_forest():
    estimator = lgb.LGBMRegressor(boosting_type="rf", n_estimators=2, random_state=1,
                                  max_depth=1, subsample=0.7, subsample_freq=1)
    fitted = utils.assert_model_predictions_match(
        utils.get_regression_model_trainer(), estimator)

    # averaging over trees is applied as a multiplication by 1/n_iter
    actual = LightGBMModelAssembler(fitted).assemble()
    assert isinstance(actual, ast.BinNumExpr)
    assert actual.op == ast.BinNumOpType.MUL


def test_regression_with_negative_values():
    estimator = lgb.LGBMRegressor(n_estimators=3, random_state=1, max_depth=1)
    utils.assert_model_predictions_match(
        utils.get_regression_w_missing_values_model_trainer(), estimator)


def test_simple_sigmoid_output_transform():
    estimator = lgb.LGBMRegressor(n_estimators=2, random_state=1, max_depth=1,
                                  objective="cross_entropy")
    fitted = utils.assert_model_predictions_match(
        utils.get_bounded_regression_model_trainer(), estimator)

    assert isinstance(LightGBMModelAssembler(fitted).assemble(), ast.SigmoidExpr)


def test_log1p_exp_output_transform():
    estimator = lgb.LGBMRegressor(n_estimators=2, random_state=1, max_depth=1,
                                  objective="cross_entropy_lambda")
    fitted = utils.assert_model_predictions_match(
        utils.get_bounded_regression_model_trainer(), estimator)

    actual = LightGBMModelAssembler(fitted).assemble()
    assert isinstance(actual, ast.Log1pExpr)
    assert isinstance(actual.expr, ast.ExpExpr)


def test_maybe_sqr_output_transform():
    estimator = lgb.LGBMRegressor(n_estimators=2, random_state=1, max_depth=1,
                                  reg_sqrt=True, objective="regression_l1")
    fitted = utils.assert_model_predictions_match(
        utils.get_regression_model_trainer(), estimator)

    # sqrt target transform is inverted as abs(x) * x
    actual = LightGBMModelAssembler(fitted).assemble()
    assert isinstance(actual, ast.BinNumExpr)
    assert actual.op == ast.BinNumOpType.MUL


def test_exp_output_transform():
    estimator = lgb.LGBMRegressor(n_estimators=2, random_state=1, max_depth=1,
                                  objective="poisson")
    fitted = utils.assert_model_predictions_match(
        utils.get_regression_model_trainer(), estimator)

    assert isinstance(LightGBMModelAssembler(fitted).assemble(), ast.ExpExpr)


def test_unknown_output_transform():
    estimator = lgb.LGBMRanker(n_estimators=1, random_state=1)
    estimator.fit(np.array([[1], [2], [3]]), np.array([1, 2, 3]), group=np.array([3]))

    assembler = LightGBMModelAssembler(estimator)

    with pytest.raises(ValueError, match="Unsupported objective function 'lambdarank'"):
        assembler.assemble()


def test_bin_class_sigmoid_output_transform():
    estimator = lgb.LGBMClassifier(n_estimators=1, random_state=1, max_depth=1, sigmoid=0.5)
    utils.get_binary_classification_model_trainer()(estimator)

    assembler = LightGBMModelAssembler(estimator)
    actual = assembler.assemble()

    sigmoid = ast.SigmoidExpr(
        ast.BinNumExpr(
            ast.NumVal(0.5),
            ast.IfExpr(
                ast.CompExpr(
                    ast.FeatureRef(20),
                    ast.NumVal(16.795),
                    ast.CompOpType.GT),
                ast.NumVal(0.5500419366076967),
                ast.NumVal(1.2782342253678096)),
            ast.BinNumOpType.MUL),
        to_reuse=True)

    expected = ast.VectorVal([
        ast.BinNumExpr(ast.NumVal(1), sigmoid, ast.BinNumOpType.SUB),
        sigmoid])

    assert utils.cmp_exprs(actual, expected)


def test_multi_class_sigmoid_output_transform():
    estimator = lgb.LGBMClassifier(n_estimators=1, random_state=1, max_depth=1, sigmoid=0.5, objective="ovr")
    estimator.fit(np.array([[1], [2], [3]]), np.array([1, 2, 3]))

    assembler = LightGBMModelAssembler(estimator)
    actual = assembler.assemble()

    sigmoid = ast.SigmoidExpr(
        ast.BinNumExpr(
            ast.NumVal(0.5),
            ast.NumVal(-1.3862943611),
            ast.BinNumOpType.MUL))

    expected = ast.VectorVal([sigmoid] * 3)

    assert utils.cmp_exprs(actual, expected)
