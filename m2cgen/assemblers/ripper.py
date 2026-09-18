import re

import numpy as np

from m2cgen import ast
from m2cgen.assemblers import utils
from m2cgen.assemblers.base import ModelAssembler

# Matches the discretized bin labels produced by wittgenstein, e.g.
# "<11.23", "11.23 - 12.5", ">23.68"
_BIN_LABEL_RE = re.compile(r"^(?:<(?P<low_open>.+)|(?P<lo>.+) - (?P<hi>.+)|>(?P<high_open>.+))$")


class WittgensteinRuleModelAssembler(ModelAssembler):
    """
    Rule-based classifiers of the ``wittgenstein`` package (RIPPER, IREP).

    The generated code outputs class probabilities, consistent with the
    ``predict_proba`` method: the weighted average of the smoothed class
    frequencies of all covering rules, or the default frequencies of the
    ruleset for samples not covered by any rule.

    Continuous features are discretized by the library into bins whose
    labels ("<a", "a - b", ">b") carry the bin edges (min exclusive,
    max inclusive). Conditions comparing against plain numeric values are
    emitted as equality comparisons; conditions on non-numeric (e.g.
    string) values are not supported.
    """

    def assemble(self):
        ruleset = self.model.ruleset_

        # The output is built as a weighted sum of constant probability
        # vectors, where each rule contributes a 0/1 indicator of whether
        # it covers the sample (mirroring wittgenstein's
        # weighted_avg_freqs), falling back to the ruleset default when no
        # rule covers the sample.
        weighted_counts = []
        weighted_freq_sums = []
        for rule in ruleset.rules:
            match = self._rule_match_expr(rule)
            freqs = np.asarray(rule.smoothed_class_freqs_, dtype=np.float64)
            weighted_counts.append(ast.IfExpr(
                match, ast.NumVal(freqs.sum()), ast.NumVal(0.0)))
            weighted_freq_sums.append(ast.IfExpr(
                match, self._vector(freqs), self._zero_vector()))

        default_freqs = np.asarray(
            ruleset.smoothed_uncovered_class_freqs_, dtype=np.float64)
        default = self._vector(default_freqs / default_freqs.sum())

        if not weighted_freq_sums:
            return default

        total_count = utils.apply_op_to_expressions(
            ast.BinNumOpType.ADD, *weighted_counts)
        freq_sum = utils.apply_op_to_expressions(
            ast.BinNumOpType.ADD, *weighted_freq_sums)
        return ast.IfExpr(
            ast.CompExpr(total_count, ast.NumVal(0.0), ast.CompOpType.GT),
            utils.apply_bin_op(
                freq_sum,
                ast.BinNumExpr(ast.NumVal(1.0), total_count, ast.BinNumOpType.DIV),
                ast.BinNumOpType.MUL),
            default)

    def _rule_match_expr(self, rule):
        # 1.0 iff every condition of the rule holds for the sample
        feature_to_idx = {
            name: idx for idx, name in enumerate(self.model.trainset_features_)}

        expr = ast.NumVal(1.0)
        for cond in rule.conds:
            for comp in self._cond_comparisons(cond, feature_to_idx):
                expr = ast.IfExpr(comp, expr, ast.NumVal(0.0))
        return expr

    @staticmethod
    def _cond_comparisons(cond, feature_to_idx):
        if cond.feature not in feature_to_idx:
            raise ValueError(
                f"Unknown feature '{cond.feature}' in rule condition")
        feature = ast.FeatureRef(feature_to_idx[cond.feature])
        label = str(cond.val).strip()

        try:
            return [ast.CompExpr(
                feature, ast.NumVal(float(label)), ast.CompOpType.EQ)]
        except ValueError:
            pass

        match = _BIN_LABEL_RE.match(label)
        if not match:
            raise NotImplementedError(
                f"Only numeric (discretized or plain numeric) conditions "
                f"are supported, got '{cond.feature}={cond.val}'")
        try:
            if match.group("low_open") is not None:
                # "<a": x <= a (the lowest bin includes its upper edge)
                return [ast.CompExpr(
                    feature, ast.NumVal(float(match.group("low_open"))),
                    ast.CompOpType.LTE)]
            if match.group("high_open") is not None:
                # ">b": x > b
                return [ast.CompExpr(
                    feature, ast.NumVal(float(match.group("high_open"))),
                    ast.CompOpType.GT)]
            # "a - b": a < x <= b
            return [
                ast.CompExpr(
                    feature, ast.NumVal(float(match.group("lo"))),
                    ast.CompOpType.GT),
                ast.CompExpr(
                    feature, ast.NumVal(float(match.group("hi"))),
                    ast.CompOpType.LTE),
            ]
        except ValueError:
            raise NotImplementedError(
                f"Only numeric (discretized or plain numeric) conditions "
                f"are supported, got '{cond.feature}={cond.val}'")

    @staticmethod
    def _vector(values):
        return ast.VectorVal([ast.NumVal(v) for v in values])

    @staticmethod
    def _zero_vector():
        return ast.VectorVal([ast.NumVal(0.0), ast.NumVal(0.0)])
