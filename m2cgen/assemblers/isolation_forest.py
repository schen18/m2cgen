import math

import numpy as np

from m2cgen import ast
from m2cgen.assemblers import utils
from m2cgen.assemblers.base import ModelAssembler


def _average_path_length(n_samples):
    """Average path length of an unsuccessful BST search in a tree of
    n_samples samples (c(n) in the isolation forest paper), matching
    sklearn's _average_path_length."""
    if n_samples <= 1:
        return 0.0
    if n_samples == 2:
        return 1.0
    return (2.0 * (math.log(n_samples - 1.0) + np.euler_gamma)
            - 2.0 * (n_samples - 1.0) / n_samples)


class IsolationForestModelAssembler(ModelAssembler):
    """
    Isolation forest of scikit-learn.

    The generated code outputs the anomaly score, consistent with the
    ``decision_function`` method: -2 ** (-mean_depth / c(max_samples))
    - offset_. Samples with a negative score are classified as anomalies
    by ``predict``.
    """

    def assemble(self):
        model = self.model
        tree_costs = [
            self._assemble_tree(estimator, features)
            for estimator, features in zip(model.estimators_,
                                           model.estimators_features_)
        ]
        depths_sum = utils.apply_op_to_expressions(
            ast.BinNumOpType.ADD, *tree_costs)

        denominator = (len(model.estimators_)
                       * _average_path_length(model.max_samples_))
        offset = -model.offset_ if hasattr(model, "offset_") else 0.5

        if denominator == 0:
            # single training sample per tree: sklearn's where-clause
            # yields depths/denominator = 1, so score_samples = -(2 ** -1)
            return ast.NumVal(-0.5 - model.offset_)

        return ast.BinNumExpr(
            ast.NumVal(offset),
            ast.PowExpr(
                ast.NumVal(2.0),
                utils.mul(ast.NumVal(-1.0 / denominator), depths_sum)),
            ast.BinNumOpType.SUB)

    def _assemble_tree(self, estimator, features):
        """Per-tree cost: depth of the leaf + c(n_node_samples) - 1."""
        tree = estimator.tree_

        def assemble_node(node_id, depth):
            if tree.children_left[node_id] == -1:
                leaf_cost = (depth + _average_path_length(tree.n_node_samples[node_id])
                             - 1.0)
                return ast.NumVal(leaf_cost)

            feature_idx = int(features[tree.feature[node_id]])
            cond = utils.lte(
                ast.FeatureRef(feature_idx),
                ast.NumVal(tree.threshold[node_id]))
            return ast.IfExpr(
                cond,
                assemble_node(tree.children_left[node_id], depth + 1),
                assemble_node(tree.children_right[node_id], depth + 1))

        return assemble_node(0, 1)
