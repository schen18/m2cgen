import numpy as np
import pytest
from sklearn import linear_model

from m2cgen import assemblers, ast

from tests import utils
from tests.utils import cmp_exprs


def test_single_feature():
    estimator = linear_model.LinearRegression()
    estimator.coef_ = np.array([1])
    estimator.intercept_ = np.array([3])

    assembler = assemblers.SklearnLinearModelAssembler(estimator)
    actual = assembler.assemble()

    expected = ast.BinNumExpr(
        ast.NumVal(3),
        ast.BinNumExpr(
            ast.FeatureRef(0),
            ast.NumVal(1),
            ast.BinNumOpType.MUL),
        ast.BinNumOpType.ADD)

    assert cmp_exprs(actual, expected)


def test_two_features():
    estimator = linear_model.LinearRegression()
    estimator.coef_ = np.array([1, 2])
    estimator.intercept_ = np.array([3])

    assembler = assemblers.SklearnLinearModelAssembler(estimator)
    actual = assembler.assemble()

    expected = ast.BinNumExpr(
        ast.BinNumExpr(
            ast.NumVal(3),
            ast.BinNumExpr(
                ast.FeatureRef(0),
                ast.NumVal(1),
                ast.BinNumOpType.MUL),
            ast.BinNumOpType.ADD),
        ast.BinNumExpr(
            ast.FeatureRef(1),
            ast.NumVal(2),
            ast.BinNumOpType.MUL),
        ast.BinNumOpType.ADD)

    assert cmp_exprs(actual, expected)


def test_multi_class():
    estimator = linear_model.LogisticRegression()
    estimator.coef_ = np.array([[1, 2], [3, 4], [5, 6]])
    estimator.intercept_ = np.array([7, 8, 9])

    assembler = assemblers.SklearnLinearModelAssembler(estimator)
    actual = assembler.assemble()

    expected = ast.VectorVal([
        ast.BinNumExpr(
            ast.BinNumExpr(
                ast.NumVal(7),
                ast.BinNumExpr(
                    ast.FeatureRef(0),
                    ast.NumVal(1),
                    ast.BinNumOpType.MUL),
                ast.BinNumOpType.ADD),
            ast.BinNumExpr(
                ast.FeatureRef(1),
                ast.NumVal(2),
                ast.BinNumOpType.MUL),
            ast.BinNumOpType.ADD),
        ast.BinNumExpr(
            ast.BinNumExpr(
                ast.NumVal(8),
                ast.BinNumExpr(
                    ast.FeatureRef(0),
                    ast.NumVal(3),
                    ast.BinNumOpType.MUL),
                ast.BinNumOpType.ADD),
            ast.BinNumExpr(
                ast.FeatureRef(1),
                ast.NumVal(4),
                ast.BinNumOpType.MUL),
            ast.BinNumOpType.ADD),
        ast.BinNumExpr(
            ast.BinNumExpr(
                ast.NumVal(9),
                ast.BinNumExpr(
                    ast.FeatureRef(0),
                    ast.NumVal(5),
                    ast.BinNumOpType.MUL),
                ast.BinNumOpType.ADD),
            ast.BinNumExpr(
                ast.FeatureRef(1),
                ast.NumVal(6),
                ast.BinNumOpType.MUL),
            ast.BinNumOpType.ADD)])

    assert cmp_exprs(actual, expected)


def test_binary_class():
    estimator = linear_model.LogisticRegression()
    estimator.coef_ = np.array([[1, 2]])
    estimator.intercept_ = np.array([3])

    assembler = assemblers.SklearnLinearModelAssembler(estimator)
    actual = assembler.assemble()

    expected = ast.BinNumExpr(
        ast.BinNumExpr(
            ast.NumVal(3),
            ast.BinNumExpr(
                ast.FeatureRef(0),
                ast.NumVal(1),
                ast.BinNumOpType.MUL),
            ast.BinNumOpType.ADD),
        ast.BinNumExpr(
            ast.FeatureRef(1),
            ast.NumVal(2),
            ast.BinNumOpType.MUL),
        ast.BinNumOpType.ADD)

    assert cmp_exprs(actual, expected)


def test_glm_identity_link_func():
    estimator = linear_model.TweedieRegressor(power=0, link="identity", max_iter=10)
    estimator = estimator.fit([[1], [2]], [0.1, 0.2])

    assembler = assemblers.SklearnGLMModelAssembler(estimator)
    actual = assembler.assemble()

    expected = ast.BinNumExpr(
        ast.NumVal(0.12),
        ast.BinNumExpr(
            ast.FeatureRef(0),
            ast.NumVal(0.02),
            ast.BinNumOpType.MUL),
        ast.BinNumOpType.ADD)

    assert cmp_exprs(actual, expected)


def test_glm_log_link_func():
    estimator = linear_model.TweedieRegressor(power=1, link="log", fit_intercept=False, max_iter=10)
    estimator = estimator.fit([[1], [2]], [0.1, 0.2])

    assembler = assemblers.SklearnGLMModelAssembler(estimator)
    actual = assembler.assemble()

    expected = ast.ExpExpr(
        ast.BinNumExpr(
            ast.NumVal(0.0),
            ast.BinNumExpr(
                ast.FeatureRef(0),
                ast.NumVal(-0.4619711397),
                ast.BinNumOpType.MUL),
            ast.BinNumOpType.ADD))

    assert cmp_exprs(actual, expected)


def test_linear_discriminant_analysis():
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    # multiclass: one linear equation per class
    estimator = LinearDiscriminantAnalysis()
    utils.assert_model_predictions_match(
        utils.get_classification_model_trainer(), estimator)


def test_linear_discriminant_analysis_binary():
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    # binary: a single signed-distance equation
    estimator = LinearDiscriminantAnalysis()
    utils.assert_model_predictions_match(
        utils.get_binary_classification_model_trainer(), estimator)


def test_glm_unknown_link_func():

    class UnknownLink:
        pass

    estimator = linear_model.TweedieRegressor(power=1, max_iter=10)
    estimator = estimator.fit([[1], [2]], [0.1, 0.2])
    # scikit-learn >= 1.1 only accepts link names as strings, so the
    # unsupported link function is injected directly into the fitted loss.
    estimator._base_loss.link = UnknownLink()

    assembler = assemblers.SklearnGLMModelAssembler(estimator)

    with pytest.raises(ValueError, match="Unsupported link function 'UnknownLink'"):
        assembler.assemble()
