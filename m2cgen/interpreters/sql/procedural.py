"""Procedural SQL interpreters (MySQL/MariaDB stored functions,
PostgreSQL PL/pgSQL functions).

Both dialects are procedural, so the standard imperative interpretation
strategy applies (variables + IF/ELSE statements); the code generator
collects variable declarations in a separate buffer since both languages
require declarations to precede statements inside the function body.
"""
from m2cgen.ast import CompOpType
from m2cgen.interpreters.code_generator import CodeTemplate, ImperativeCodeGenerator
from m2cgen.interpreters.interpreter import ImperativeToCodeInterpreter
from m2cgen.interpreters.mixins import LinearAlgebraMixin, PowExprFunctionMixin
from m2cgen.interpreters.sql.base import SqlFeatureRefMixin


class ProceduralCodeGenerator(ImperativeCodeGenerator):

    scalar_type = NotImplemented
    vector_type = NotImplemented

    tpl_num_value = CodeTemplate("{value}")
    tpl_infix_expression = CodeTemplate("{left} {op} {right}")

    def reset_state(self):
        super().reset_state()
        self._declarations = []

    def add_var_declaration(self, size):
        var_name = self.get_var_name()
        var_type = self.vector_type if size > 1 else self.scalar_type
        self._declarations.append(
            self.tpl_var_declaration(var_type=var_type, var_name=var_name))
        return var_name

    def _comp_op_overwrite(self, op):
        if op == CompOpType.EQ:
            return "="
        return op.value

    def _quote_ident(self, name):
        raise NotImplementedError

    def finalize_function(self, name, param_names, returns_vector):
        """Assembles the full function definition: header, declaration
        section (collected separately), the body statements from the code
        buffer and the footer."""
        body = self._code_buf.getvalue()
        self._finalize_buffer()

        pad = " " * self._indent
        body_lines = "".join(
            pad + line + "\n" if line else ""
            for line in body.split("\n")).rstrip("\n")

        return self._function_template()(
            name=self._quote_ident(name),
            params=self._render_params(param_names),
            return_type=self.vector_type if returns_vector else self.scalar_type,
            declarations=self._render_declarations(),
            body=body_lines)

    def _render_params(self, param_names):
        # param names arrive already quoted by the interpreter
        return ", ".join(f"{p} {self.scalar_type}" for p in param_names)

    def _render_declarations(self):
        if not self._declarations:
            return ""
        return "\n".join(" " * self._indent + d for d in self._declarations)

    def _function_template(self):
        raise NotImplementedError


class ProceduralSqlInterpreter(SqlFeatureRefMixin,
                               ImperativeToCodeInterpreter,
                               PowExprFunctionMixin,
                               LinearAlgebraMixin):
    """Common behavior of the procedural SQL interpreters."""

    def __init__(self, cg, indent=4, function_name="score", feature_names=None):
        self.function_name = function_name
        super().__init__(cg, feature_names=feature_names)

    def interpret(self, expr):
        self._cg.reset_state()
        self._reset_reused_expr_cache()
        self._reset_sql_state()

        last_result = self._do_interpret(expr)
        self._cg.add_return_statement(last_result)

        param_names = [
            self._get_feature_ident(i)
            for i in range(self._max_feature_index + 1)]

        code = self._cg.finalize_function(
            self.function_name, param_names, expr.output_size > 1)

        if self.with_linear_algebra:
            code = self._linear_algebra_code() + "\n\n" + code

        return code

    def _render_feature_ident(self, index, name):
        return self._cg._quote_ident(name)

    def _linear_algebra_code(self):
        raise NotImplementedError
