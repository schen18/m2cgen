import json

import numpy as np
import pytest
import xgboost as xgb

from m2cgen import ast, export_to_python
from m2cgen.assemblers import XGBoostModelAssemblerSelector
from m2cgen.assemblers.boosting import XGBoostTreeModelAssembler

from tests import utils

# XGBoost casts inputs to float32 before comparing them against (float32)
# tree thresholds, while the generated code compares the original float64
# inputs against the dumped thresholds. For samples whose feature values lie
# in the tiny gap between the two representations the generated code can
# therefore follow a different branch of a tree. This is a documented
# limitation (see the FAQ in README), so a small fraction of samples is
# allowed to diverge.
MIN_MATCH_FRACTION = 0.9


def _assert_predictions_match(model, X, expected, rtol=1e-5, atol=1e-8,
                              min_match=1.0):
    scope = {}
    exec(export_to_python(model), scope)
    score = scope["score"]

    matched = 0
    for idx in range(len(X)):
        actual = np.atleast_1d(score(X[idx].tolist()))
        exp = np.atleast_1d(expected[idx])
        if actual.shape == exp.shape and np.allclose(actual, exp, rtol=rtol, atol=atol):
            matched += 1

    assert matched >= min_match * len(X), (
        f"Only {matched}/{len(X)} samples match")


def test_binary_classification():
    estimator = xgb.XGBClassifier(n_estimators=2, random_state=1, max_depth=1)
    trainer = utils.get_classification_binary_random_data_model_trainer()
    _, __, estimator = trainer(estimator)

    actual = XGBoostModelAssemblerSelector(estimator).assemble()

    assert isinstance(actual, ast.VectorVal)
    assert len(actual.exprs) == 2
    assert isinstance(actual.exprs[1], ast.SigmoidExpr)

    trainer = utils.ModelTrainer.get_instance("classification_binary_rnd")
    _assert_predictions_match(
        estimator, trainer.X, estimator.predict_proba(trainer.X),
        min_match=MIN_MATCH_FRACTION)


def test_multi_class():
    estimator = xgb.XGBClassifier(n_estimators=1, random_state=1, max_depth=1)
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([0, 1, 2])
    estimator.fit(X, y)

    actual = XGBoostModelAssemblerSelector(estimator).assemble()

    assert isinstance(actual, ast.SoftmaxExpr)
    assert len(actual.exprs) == 3

    _assert_predictions_match(
        estimator, X, estimator.predict_proba(X), min_match=MIN_MATCH_FRACTION)


def test_regression():
    # default (automatically estimated) base score: only available from the
    # booster configuration since XGBoost 1.7
    estimator = xgb.XGBRegressor(n_estimators=2, random_state=1, max_depth=1)
    trainer = utils.get_regression_random_data_model_trainer()
    _, __, estimator = trainer(estimator)

    XGBoostModelAssemblerSelector(estimator).assemble()

    trainer = utils.ModelTrainer.get_instance("regression_rnd")
    _assert_predictions_match(
        estimator, trainer.X, estimator.predict(trainer.X),
        min_match=MIN_MATCH_FRACTION)


def test_regression_explicit_base_score():
    # explicitly specified base score is still taken from get_params()
    estimator = xgb.XGBRegressor(
        n_estimators=2, random_state=1, max_depth=1, base_score=0.6)
    trainer = utils.get_regression_random_data_model_trainer()
    _, __, estimator = trainer(estimator)

    XGBoostModelAssemblerSelector(estimator).assemble()

    trainer = utils.ModelTrainer.get_instance("regression_rnd")
    _assert_predictions_match(
        estimator, trainer.X, estimator.predict(trainer.X),
        min_match=MIN_MATCH_FRACTION)


def test_regression_early_stopping():
    # with early stopping only trees up to best_iteration (inclusive) must
    # be used; best_ntree_limit was removed in XGBoost 2.0
    estimator = xgb.XGBRegressor(
        n_estimators=100, max_depth=2, random_state=1,
        early_stopping_rounds=5)
    trainer = utils.get_regression_random_data_model_trainer()
    estimator.fit(trainer.X_train, trainer.y_train,
                  eval_set=[(trainer.X_test, trainer.y_test)], verbose=False)

    assert estimator.best_iteration is not None

    XGBoostModelAssemblerSelector(estimator).assemble()

    _assert_predictions_match(
        estimator, trainer.X, estimator.predict(trainer.X),
        min_match=MIN_MATCH_FRACTION)


def test_multi_class_early_stopping():
    estimator = xgb.XGBClassifier(
        n_estimators=100, max_depth=2, random_state=1,
        early_stopping_rounds=5)
    trainer = utils.get_classification_random_data_model_trainer()
    estimator.fit(trainer.X_train, trainer.y_train,
                  eval_set=[(trainer.X_test, trainer.y_test)], verbose=False)

    assert estimator.best_iteration is not None

    XGBoostModelAssemblerSelector(estimator).assemble()

    _assert_predictions_match(
        estimator, trainer.X, estimator.predict_proba(trainer.X),
        min_match=MIN_MATCH_FRACTION)


def test_regression_saved_without_feature_names():
    estimator = xgb.XGBRegressor(n_estimators=2, random_state=1, max_depth=1)
    trainer = utils.get_regression_random_data_model_trainer()
    _, __, estimator = trainer(estimator)

    with utils.tmp_dir() as tmp_dirpath:
        filename = tmp_dirpath / "tmp.file"
        estimator.save_model(filename)
        estimator = xgb.XGBRegressor()
        estimator.load_model(filename)

    XGBoostModelAssemblerSelector(estimator).assemble()

    trainer = utils.ModelTrainer.get_instance("regression_rnd")
    _assert_predictions_match(
        estimator, trainer.X, estimator.predict(trainer.X),
        min_match=MIN_MATCH_FRACTION)


def test_linear_model():
    # Default updater ("shotgun") is nondeterministic
    estimator = xgb.XGBRegressor(n_estimators=2, random_state=1, updater="coord_descent",
                                 feature_selector="shuffle", booster="gblinear")
    trainer = utils.get_regression_random_data_model_trainer()
    _, __, estimator = trainer(estimator)

    XGBoostModelAssemblerSelector(estimator).assemble()

    # weights/bias are dumped by XGBoost with limited precision
    trainer = utils.ModelTrainer.get_instance("regression_rnd")
    _assert_predictions_match(
        estimator, trainer.X, estimator.predict(trainer.X),
        rtol=1e-4, atol=1e-3)


def test_linear_model_multi_class():
    estimator = xgb.XGBClassifier(n_estimators=20, random_state=1,
                                  booster="gblinear")
    trainer = utils.get_classification_random_data_model_trainer()
    _, __, estimator = trainer(estimator)

    XGBoostModelAssemblerSelector(estimator).assemble()

    trainer = utils.ModelTrainer.get_instance("classification_rnd")
    _assert_predictions_match(
        estimator, trainer.X, estimator.predict_proba(trainer.X),
        rtol=1e-4, atol=1e-3)


def test_regression_random_forest():
    estimator = xgb.XGBRFRegressor(n_estimators=2, random_state=1, max_depth=1)
    trainer = utils.get_regression_random_data_model_trainer()
    _, __, estimator = trainer(estimator)

    XGBoostModelAssemblerSelector(estimator).assemble()

    trainer = utils.ModelTrainer.get_instance("regression_rnd")
    _assert_predictions_match(
        estimator, trainer.X, estimator.predict(trainer.X),
        min_match=MIN_MATCH_FRACTION)


class _FakeBooster:

    def __init__(self, dump):
        self._dump = dump
        self.feature_names = None

    def get_dump(self, dump_format="json"):
        return [self._dump]

    def save_config(self):
        return json.dumps({"learner": {
            "learner_model_param": {"base_score": "[0.5]"},
            "gradient_booster": {"gbtree_model_param": {
                "num_parallel_tree": "1", "num_trees": "1"}}}})


class _FakeModel:

    def __init__(self, dump):
        self._booster = _FakeBooster(dump)

    def get_params(self):
        return {}

    def get_booster(self):
        return self._booster


def test_categorical_splits_not_supported():
    tree = json.dumps({
        "nodeid": 0, "split": "f0", "split_condition": [1, 3],
        "yes": 1, "no": 2, "missing": 2,
        "children": [
            {"nodeid": 1, "leaf": 0.1},
            {"nodeid": 2, "leaf": 0.2}]})

    assembler = XGBoostTreeModelAssembler(_FakeModel(tree))

    with pytest.raises(NotImplementedError, match="[Cc]ategorical splits"):
        assembler.assemble()


def test_vector_leaves_not_supported():
    tree = json.dumps({"nodeid": 0, "leaf": [0.1, 0.2]})

    assembler = XGBoostTreeModelAssembler(_FakeModel(tree))

    with pytest.raises(NotImplementedError, match="[Vv]ector leaves"):
        assembler.assemble()
