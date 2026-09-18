import json
import math

import numpy as np

from m2cgen import ast
from m2cgen.assemblers import utils
from m2cgen.assemblers.base import ModelAssembler
from m2cgen.assemblers.linear import _linear_to_ast


class BaseBoostingAssembler(ModelAssembler):

    classifier_names = {}
    multiclass_params_seq_len = 1

    def __init__(self, model, estimator_params, base_score=0.0):
        super().__init__(model)
        self._all_estimator_params = estimator_params
        # Either a scalar or one base score per output (multi-class models
        # of XGBoost >= 3.1 store one value per class).
        self._base_scores = (
            [base_score] if isinstance(base_score, (int, float)) else list(base_score))

        self._output_size = 1
        self._is_classification = False

        model_class_name = type(model).__name__
        if model_class_name in self.classifier_names:
            self._is_classification = True
            if model.n_classes_ > 2:
                self._output_size = model.n_classes_

    def assemble(self):
        if self._is_classification:
            if self._output_size == 1:
                return self._assemble_bin_class_output(self._all_estimator_params)
            else:
                return self._assemble_multi_class_output(self._all_estimator_params)
        else:
            result_ast = self._assemble_single_output(
                self._all_estimator_params, base_score=self._base_scores[0])
            return self._single_convert_output(result_ast)

    def _assemble_single_output(self, estimator_params, base_score=0.0, split_idx=0):
        estimators_ast = self._assemble_estimators(estimator_params, split_idx)

        tmp_ast = utils.apply_op_to_expressions(
            ast.BinNumOpType.ADD,
            *estimators_ast)

        if base_score != 0.0:
            tmp_ast = utils.apply_bin_op(
                ast.NumVal(base_score),
                tmp_ast,
                ast.BinNumOpType.ADD)

        result_ast = self._final_transform(tmp_ast)

        return result_ast

    def _assemble_multi_class_output(self, estimator_params):
        # Multi-class output is calculated based on discussion in
        # https://github.com/dmlc/xgboost/issues/1746#issuecomment-295962863
        # and the enhancement to support boosted forests in XGBoost.
        splits = _split_estimator_params_by_classes(
            estimator_params, self._output_size,
            self.multiclass_params_seq_len)

        exprs = [
            self._assemble_single_output(
                e,
                # one base score per class for XGBoost >= 3.1,
                # a single shared value otherwise
                base_score=self._base_scores[i] if len(self._base_scores) > 1
                else self._base_scores[0],
                split_idx=i)
            for i, e in enumerate(splits)
        ]

        return self._multi_class_convert_output(exprs)

    def _assemble_bin_class_output(self, estimator_params):
        # Base score is calculated based on
        # https://github.com/dmlc/xgboost/blob/8de7f1928e4815843fbf8773a5ac7ecbc37b2e15/src/objective/regression_loss.h#L91
        # return -logf(1.0f / base_score - 1.0f);
        raw_base_score = self._base_scores[0]
        base_score = 0.0
        if raw_base_score != 0.0:
            if not 0.0 < raw_base_score < 1.0:
                raise ValueError(
                    f"Unexpected base score '{raw_base_score}': expected a "
                    f"probability in the (0, 1) interval")
            base_score = -math.log(1.0 / raw_base_score - 1.0)

        expr = self._assemble_single_output(estimator_params, base_score=base_score)

        proba_expr = self._bin_class_convert_output(expr)

        return ast.VectorVal([
            ast.BinNumExpr(ast.NumVal(1.0), proba_expr, ast.BinNumOpType.SUB),
            proba_expr
        ])

    def _final_transform(self, ast_to_transform):
        return ast_to_transform

    def _multi_class_convert_output(self, exprs):
        return ast.SoftmaxExpr(exprs)

    def _bin_class_convert_output(self, expr, to_reuse=True):
        return ast.SigmoidExpr(expr, to_reuse=to_reuse)

    def _single_convert_output(self, expr):
        return expr

    def _assemble_estimators(self, estimator_params, split_idx):
        raise NotImplementedError


class BaseTreeBoostingAssembler(BaseBoostingAssembler):

    def __init__(self, model, trees, base_score=0.0, tree_limit=None):
        super().__init__(model, trees, base_score=base_score)
        assert tree_limit is None or tree_limit > 0, "Unexpected tree limit"
        self._tree_limit = tree_limit

    def _assemble_estimators(self, trees, split_idx):
        if self._tree_limit:
            trees = trees[:self._tree_limit]

        return [self._assemble_tree(t) for t in trees]

    def _assemble_tree(self, tree):
        raise NotImplementedError


def _get_xgboost_booster_config(model):
    return json.loads(model.get_booster().save_config())


def _get_xgboost_base_score(model):
    """Returns the list of base scores (one per output group) of a fitted
    model.

    XGBoost >= 1.7 no longer fills get_params() from the fitted model, so the
    value is read from the booster configuration, where it is stored as a
    JSON value: a scalar for single-output models and one entry per class
    for multi-class models (XGBoost >= 3.1). For binary classification the
    stored value is in the probability space; for multi-class (>= 3.2) and
    regression models it is in the margin space.
    """
    base_score = model.get_params().get("base_score")
    if isinstance(base_score, (int, float)):
        # XGBoost <= 1.6 filled get_params() from the fitted model.
        return [float(base_score)]

    raw = json.loads(
        _get_xgboost_booster_config(model)["learner"]["learner_model_param"]["base_score"])
    if isinstance(raw, (int, float)):
        return [float(raw)]
    return [float(v) for v in raw]


def _get_xgboost_num_parallel_tree(model):
    try:
        return int(_get_xgboost_booster_config(model)["learner"]["gradient_booster"]
                   ["gbtree_model_param"]["num_parallel_tree"])
    except (KeyError, ValueError):
        return 1


class XGBoostTreeModelAssembler(BaseTreeBoostingAssembler):

    classifier_names = {
        "XGBClassifier",
        "XGBRFClassifier"
    }

    def __init__(self, model):
        self.multiclass_params_seq_len = _get_xgboost_num_parallel_tree(model)
        feature_names = model.get_booster().feature_names
        self._feature_name_to_idx = {
            name: idx for idx, name in enumerate(feature_names or [])
        }

        model_dump = model.get_booster().get_dump(dump_format="json")
        trees = [json.loads(d) for d in model_dump]

        # Limit the number of trees that should be used for assembling (if
        # applicable). Estimators use iterations in the [0, best_iteration]
        # range for predictions when early stopping is enabled.
        # (best_ntree_limit was removed in XGBoost 2.0.)
        best_iteration = getattr(model, "best_iteration", None)
        tree_limit = None
        if best_iteration is not None:
            trees_per_round = self.multiclass_params_seq_len
            if type(model).__name__ in self.classifier_names and model.n_classes_ > 2:
                trees_per_round *= model.n_classes_
            tree_limit = (best_iteration + 1) * trees_per_round

        # The limit is applied to the flat tree list up-front: applying it
        # later (per output, after the trees are split by classes) would
        # truncate the wrong trees for multi-class models.
        if tree_limit is not None:
            trees = trees[:tree_limit]

        super().__init__(model,
                         trees,
                         base_score=_get_xgboost_base_score(model))

    def _assemble_tree(self, tree):
        if "leaf" in tree:
            if isinstance(tree["leaf"], list):
                raise NotImplementedError(
                    "Multi-output trees with vector leaves "
                    "(multi_strategy='multi_output_tree') are not supported")
            # The dumped values are kept as float64: rounding them to the
            # float32 representation would produce literals which do not
            # round-trip in the generated code and shift tree thresholds.
            return ast.NumVal(tree["leaf"])

        if isinstance(tree["split_condition"], list):
            raise NotImplementedError(
                "Categorical splits are not supported")

        threshold = ast.NumVal(tree["split_condition"])
        split = tree["split"]
        if split in self._feature_name_to_idx:
            feature_idx = self._feature_name_to_idx[split]
        elif split[0] == "f":
            feature_idx = int(split[1:])
        else:
            feature_idx = int(split)
        feature_ref = ast.FeatureRef(feature_idx)

        # Since comparison with NaN (missing) value always returns false we
        # should make sure that the node ID specified in the "missing" field
        # always ends up in the "else" branch of the ast.IfExpr.
        use_lt_comp = tree["missing"] == tree["no"]
        if use_lt_comp:
            comp_op = ast.CompOpType.LT
            true_child_id = tree["yes"]
            false_child_id = tree["no"]
        else:
            comp_op = ast.CompOpType.GTE
            true_child_id = tree["no"]
            false_child_id = tree["yes"]

        return ast.IfExpr(
            ast.CompExpr(feature_ref, threshold, comp_op),
            self._assemble_child_tree(tree, true_child_id),
            self._assemble_child_tree(tree, false_child_id))

    def _assemble_child_tree(self, tree, child_id):
        for child in tree["children"]:
            if child["nodeid"] == child_id:
                return self._assemble_tree(child)
        assert False, f"Unexpected child ID: {child_id}"


class XGBoostLinearModelAssembler(BaseBoostingAssembler):

    classifier_names = {"XGBClassifier"}

    def __init__(self, model):
        model_dump = model.get_booster().get_dump(dump_format="json")
        weights = utils.to_2d_array(
            json.loads(model_dump[0])["weight"]).reshape(1, -1)
        self._bias = utils.to_1d_array(json.loads(model_dump[0])["bias"])

        if (type(model).__name__ in self.classifier_names
                and model.n_classes_ == 2):
            raise NotImplementedError(
                "XGBClassifier with the gblinear booster is not supported "
                "for binary classification (the gblinear booster is "
                "deprecated in XGBoost)")

        if len(self._bias) > 1:
            # For multi-class models the weights are stored feature-major
            # (feature0_class0, feature0_class1, ...) and the bias values
            # already include the base score.
            weights = weights.reshape(-1, len(self._bias)).T
            base_score = 0.0
        else:
            base_score = _get_xgboost_base_score(model)

        super().__init__(model, weights, base_score=base_score)

    def _assemble_estimators(self, estimator_params, split_idx):
        # estimator_params holds a single per-output weight vector (either
        # the whole model for single-output, or the per-class row selected
        # by _split_estimator_params_by_classes).
        coef = utils.to_1d_array(estimator_params[0])
        return [_linear_to_ast(coef, self._bias[split_idx])]


class XGBoostModelAssemblerSelector(ModelAssembler):

    def __init__(self, model, *args, **kwargs):
        model_dump = model.get_booster().get_dump(dump_format="json")
        if len(model_dump) == 1 and all(i in json.loads(model_dump[0]) for i in ("weight", "bias")):
            self.assembler = XGBoostLinearModelAssembler(model)
        else:
            self.assembler = XGBoostTreeModelAssembler(model, *args, **kwargs)

    def assemble(self):
        return self.assembler.assemble()


class LightGBMModelAssembler(BaseTreeBoostingAssembler):

    classifier_names = {"LGBMClassifier"}

    def __init__(self, model):
        model_dump = model.booster_.dump_model()
        trees = [m["tree_structure"] for m in model_dump["tree_info"]]

        self.n_iter = len(trees) // model_dump["num_tree_per_iteration"]
        self.average_output = model_dump.get("average_output", False)
        self.objective_config_parts = model_dump.get("objective", "custom").split(" ")
        self.objective_name = self.objective_config_parts[0]

        super().__init__(model, trees)

        # One-vs-all models have one tree group per class with a sigmoid
        # transform applied per class, also when there are only two classes.
        if (self.objective_name == "multiclassova"
                and self._is_classification
                and self._output_size == 1):
            self._output_size = model.n_classes_

    def _final_transform(self, ast_to_transform):
        if self.average_output:
            coef = 1 / self.n_iter
            return utils.apply_bin_op(
                ast_to_transform,
                ast.NumVal(coef),
                ast.BinNumOpType.MUL)
        else:
            return super()._final_transform(ast_to_transform)

    def _multi_class_convert_output(self, exprs):
        supported_objectives = {
            "multiclass": super()._multi_class_convert_output,
            "multiclassova": self._multi_class_sigmoid_transform,
            "custom": super()._single_convert_output,
        }
        if self.objective_name not in supported_objectives:
            raise ValueError(f"Unsupported objective function '{self.objective_name}'")
        return supported_objectives[self.objective_name](exprs)

    def _multi_class_sigmoid_transform(self, exprs):
        return ast.VectorVal([
            self._bin_class_sigmoid_transform(expr, to_reuse=False)
            for expr in exprs
        ])

    def _bin_class_convert_output(self, expr, to_reuse=True):
        supported_objectives = {
            "binary": self._bin_class_sigmoid_transform,
            # one-vs-all sigmoid is applied per class, so for binary models
            # it is equivalent to the regular sigmoid transform
            "multiclassova": self._bin_class_sigmoid_transform,
            "custom": super()._single_convert_output,
        }
        if self.objective_name not in supported_objectives:
            raise ValueError(f"Unsupported objective function '{self.objective_name}'")
        return supported_objectives[self.objective_name](expr)

    def _bin_class_sigmoid_transform(self, expr, to_reuse=True):
        coef = 1.0
        for config_part in self.objective_config_parts:
            config_entry = config_part.split(":")
            if config_entry[0] == "sigmoid":
                coef = np.float64(config_entry[1])
                break
        return super()._bin_class_convert_output(
            utils.mul(ast.NumVal(coef), expr) if coef != 1.0 else expr,
            to_reuse=to_reuse)

    def _single_convert_output(self, expr):
        supported_objectives = {
            "cross_entropy": ast.SigmoidExpr,
            "cross_entropy_lambda": self._log1p_exp_transform,
            "regression": self._maybe_sqr_transform,
            "regression_l1": self._maybe_sqr_transform,
            "huber": super()._single_convert_output,
            "fair": self._maybe_sqr_transform,
            "poisson": self._exp_transform,
            "quantile": self._maybe_sqr_transform,
            "mape": self._maybe_sqr_transform,
            "gamma": self._exp_transform,
            "tweedie": self._exp_transform,
            "custom": super()._single_convert_output,
        }
        if self.objective_name not in supported_objectives:
            raise ValueError(
                f"Unsupported objective function '{self.objective_name}'")
        return supported_objectives[self.objective_name](expr)

    def _log1p_exp_transform(self, expr):
        return ast.Log1pExpr(ast.ExpExpr(expr))

    def _maybe_sqr_transform(self, expr):
        if "sqrt" in self.objective_config_parts:
            expr = ast.IdExpr(expr, to_reuse=True)
            return utils.mul(ast.AbsExpr(expr), expr)
        else:
            return expr

    def _exp_transform(self, expr):
        return ast.ExpExpr(expr)

    def _assemble_tree(self, tree):
        if "leaf_value" in tree:
            return ast.NumVal(tree["leaf_value"])

        threshold = ast.NumVal(tree["threshold"])
        feature_ref = ast.FeatureRef(tree["split_feature"])

        op = ast.CompOpType.from_str_op(tree["decision_type"])
        assert op == ast.CompOpType.LTE, "Unexpected comparison op"

        missing_type = tree['missing_type']

        if missing_type not in {"NaN", "None"}:
            raise ValueError(f"Unknown missing_type: {missing_type}")

        reverse_condition = missing_type == "NaN" and tree["default_left"]
        reverse_condition |= missing_type == "None" and tree["threshold"] >= 0
        if reverse_condition:
            op = ast.CompOpType.GT
            true_child = tree["right_child"]
            false_child = tree["left_child"]
        else:
            true_child = tree["left_child"]
            false_child = tree["right_child"]

        return ast.IfExpr(
            ast.CompExpr(feature_ref, threshold, op),
            self._assemble_tree(true_child),
            self._assemble_tree(false_child))


def _split_estimator_params_by_classes(values, n_classes, params_seq_len):
    # Splits are computed based on a comment
    # https://github.com/dmlc/xgboost/issues/1746#issuecomment-267400592
    # and the enhancement to support boosted forests in XGBoost.
    values_len = len(values)
    block_len = n_classes * params_seq_len
    indices = list(range(values_len))
    indices_by_class = np.array(
        [[indices[i:i + params_seq_len]
          for i in range(j, values_len, block_len)]
         for j in range(0, block_len, params_seq_len)]
        ).reshape(n_classes, -1)
    return [[values[idx] for idx in class_idxs] for class_idxs in indices_by_class]
