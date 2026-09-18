from m2cgen.ast import BinNumOpType
from m2cgen.interpreters.code_generator import CodeTemplate
from m2cgen.interpreters.sql.procedural import ProceduralCodeGenerator, ProceduralSqlInterpreter


class PostgresCodeGenerator(ProceduralCodeGenerator):

    scalar_type = "double precision"
    vector_type = "double precision[]"

    tpl_var_declaration = CodeTemplate("{var_name} {var_type};")
    tpl_return_statement = CodeTemplate("RETURN {value};")
    tpl_if_statement = CodeTemplate("IF {if_def} THEN")
    tpl_else_statement = CodeTemplate("ELSE")
    tpl_block_termination = CodeTemplate("END IF;")
    tpl_var_assignment = CodeTemplate("{var_name} := {value};")

    def _quote_ident(self, name):
        return f'"{name}"'

    def vector_init(self, values):
        return f"ARRAY[{', '.join(values)}]"

    def _function_template(self):
        return CodeTemplate(
            "CREATE FUNCTION {name}({params}) RETURNS {return_type}\n"
            "LANGUAGE plpgsql IMMUTABLE AS $$\n"
            "DECLARE\n"
            "{declarations}\n"
            "BEGIN\n"
            "{body}\n"
            "END\n"
            "$$;")


LINEAR_ALGEBRA_CODE = """
CREATE FUNCTION m2c_add_vectors(a double precision[], b double precision[])
RETURNS double precision[] LANGUAGE sql IMMUTABLE AS $$
    SELECT array_agg(a[i] + b[i] ORDER BY i)
    FROM generate_subscripts(a, 1) AS i
$$;

CREATE FUNCTION m2c_mul_vector_number(a double precision[], n double precision)
RETURNS double precision[] LANGUAGE sql IMMUTABLE AS $$
    SELECT array_agg(v * n ORDER BY i)
    FROM unnest(a) WITH ORDINALITY AS t(v, i)
$$;
""".strip()


class PostgresInterpreter(ProceduralSqlInterpreter):

    # PostgreSQL does not provide tanh/log1p natively: those are computed
    # via the AST-level fallback expressions.
    abs_function_name = "ABS"
    atan_function_name = "ATAN"
    exponent_function_name = "EXP"
    logarithm_function_name = "LN"
    log1p_function_name = NotImplemented
    power_function_name = "POWER"
    sigmoid_function_name = NotImplemented
    softmax_function_name = NotImplemented
    sqrt_function_name = "SQRT"
    tanh_function_name = NotImplemented

    supported_bin_vector_ops = {
        BinNumOpType.ADD: "m2c_add_vectors",
    }
    supported_bin_vector_num_ops = {
        BinNumOpType.MUL: "m2c_mul_vector_number",
    }

    def __init__(self, indent=4, function_name="score", feature_names=None):
        super().__init__(
            PostgresCodeGenerator(indent=indent),
            indent=indent, function_name=function_name,
            feature_names=feature_names)

    def _linear_algebra_code(self):
        return LINEAR_ALGEBRA_CODE
