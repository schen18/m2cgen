"""DuckDB interpreter: emits a `CREATE OR REPLACE MACRO` with a single
expression. Variables (reused sub-expressions) are linearized as a chain
of CTEs inside a scalar subquery, keeping expression nesting shallow for
arbitrarily large models.
"""
from m2cgen.ast import BinNumOpType, CompOpType
from m2cgen.interpreters.code_generator import BaseCodeGenerator, CodeTemplate
from m2cgen.interpreters.interpreter import ToCodeInterpreter
from m2cgen.interpreters.mixins import PowExprFunctionMixin
from m2cgen.interpreters.sql.base import SqlFeatureRefMixin
from m2cgen.interpreters.utils import CachedResult, format_float


class DuckDbCodeGenerator(BaseCodeGenerator):

    tpl_num_value = CodeTemplate("{value}")
    tpl_infix_expression = CodeTemplate("{left} {op} {right}")
    tpl_array_index_access = CodeTemplate("{array_name}[{index}]")

    def _comp_op_overwrite(self, op):
        if op == CompOpType.EQ:
            return "="
        return op.value


class DuckDbInterpreter(SqlFeatureRefMixin, PowExprFunctionMixin,
                        ToCodeInterpreter):

    # DuckDB natively provides all of these (including tanh); the ones
    # which are left as NotImplemented use the AST-level fallbacks.
    abs_function_name = "abs"
    atan_function_name = "atan"
    exponent_function_name = "exp"
    logarithm_function_name = "ln"
    log1p_function_name = NotImplemented
    power_function_name = "power"
    sigmoid_function_name = NotImplemented
    softmax_function_name = NotImplemented
    sqrt_function_name = "sqrt"
    tanh_function_name = "tanh"

    def __init__(self, indent=4, function_name="score", feature_names=None):
        self.function_name = function_name
        self.indent = indent
        self._cte_items = []
        super().__init__(
            DuckDbCodeGenerator(), feature_names=feature_names)

    def interpret(self, expr):
        self._cte_items = []
        self._reset_reused_expr_cache()
        self._reset_sql_state()

        result = self._do_interpret(expr)

        params = ", ".join(
            self._get_feature_ident(i)
            for i in range(self._max_feature_index + 1))

        pad = " " * self.indent
        body_lines = []
        if self._cte_items:
            ctes = ",\n".join(
                f"{pad}{pad}{name} AS (SELECT {expr_sql} AS x)"
                for name, expr_sql in self._cte_items)
            body_lines.append(f"{pad}WITH")
            body_lines.append(ctes)
        body_lines.append(f"{pad}SELECT {result}")

        return (
            f"CREATE OR REPLACE MACRO {self.function_name}({params}) AS (\n"
            + "\n".join(body_lines)
            + "\n);")

    def interpret_if_expr(self, expr, **kwargs):
        test = self._do_interpret(expr.test, **kwargs)
        body = self._do_interpret(expr.body, **kwargs)
        orelse = self._do_interpret(expr.orelse, **kwargs)
        return f"CASE WHEN {test} THEN {body} ELSE {orelse} END"

    def interpret_vector_val(self, expr, **kwargs):
        nested = [self._do_interpret(e, **kwargs) for e in expr.exprs]
        return f"[{', '.join(nested)}]"

    def interpret_bin_vector_expr(self, expr, **kwargs):
        if expr.op != BinNumOpType.ADD:
            raise NotImplementedError(f"Op '{expr.op.name}' is unsupported")
        left = self._do_interpret(expr.left, **kwargs)
        right = self._do_interpret(expr.right, **kwargs)
        return (
            f"list_transform(list_zip({left}, {right}), t -> t[1] + t[2])")

    def interpret_bin_vector_num_expr(self, expr, **kwargs):
        if expr.op != BinNumOpType.MUL:
            raise NotImplementedError(f"Op '{expr.op.name}' is unsupported")
        left = self._do_interpret(expr.left, **kwargs)
        num = self._do_interpret(expr.right, **kwargs)
        return f"list_transform({left}, x -> x * {num})"

    def _cache_reused_expr(self, expr, expr_result):
        name = f"v{len(self._cte_items)}"
        self._cte_items.append((name, expr_result))
        result = f"(SELECT x FROM {name})"
        self._cached_expr_results[expr] = CachedResult(
            var_name=result, expr_result=None)
        return result

    def _render_feature_ident(self, index, name):
        # macro parameters cannot be quoted identifiers in DuckDB
        return name

    # numeric literals are emitted in scientific notation so that they are
    # parsed as DOUBLE instead of DECIMAL (exact-decimal arithmetic would
    # produce slightly different results than the float64 semantics of the
    # original model)
    def interpret_num_val(self, expr, **kwargs):
        formatted = format_float(expr.value)
        if "e" not in formatted and "E" not in formatted:
            formatted += "E0"
        return formatted
