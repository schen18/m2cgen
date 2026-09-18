from m2cgen.ast import BinNumOpType
from m2cgen.interpreters.code_generator import CodeTemplate
from m2cgen.interpreters.sql.procedural import ProceduralCodeGenerator, ProceduralSqlInterpreter


class MySqlCodeGenerator(ProceduralCodeGenerator):

    scalar_type = "DOUBLE"
    vector_type = "JSON"

    tpl_var_declaration = CodeTemplate("DECLARE {var_name} {var_type};")
    tpl_return_statement = CodeTemplate("RETURN {value};")
    tpl_if_statement = CodeTemplate("IF {if_def} THEN")
    tpl_else_statement = CodeTemplate("ELSE")
    tpl_block_termination = CodeTemplate("END IF;")
    tpl_var_assignment = CodeTemplate("SET {var_name} = {value};")

    def _quote_ident(self, name):
        return f"`{name}`"

    def vector_init(self, values):
        return f"JSON_ARRAY({', '.join(values)})"

    def _function_template(self):
        return CodeTemplate(
            "CREATE FUNCTION {name}({params}) RETURNS {return_type} DETERMINISTIC\n"
            "BEGIN\n"
            "{declarations}\n"
            "\n"
            "{body}\n"
            "END")


LINEAR_ALGEBRA_CODE = """
CREATE FUNCTION `m2c_add_vectors`(a JSON, b JSON) RETURNS JSON DETERMINISTIC
BEGIN
    DECLARE i INT DEFAULT 0;
    DECLARE res JSON;
    SET res = JSON_ARRAY();
    WHILE i < JSON_LENGTH(a) DO
        SET res = JSON_ARRAYAPPEND(res, '$',
            CAST(JSON_EXTRACT(a, CONCAT('$[', i, ']')) AS DOUBLE) +
            CAST(JSON_EXTRACT(b, CONCAT('$[', i, ']')) AS DOUBLE));
        SET i = i + 1;
    END WHILE;
    RETURN res;
END;

CREATE FUNCTION `m2c_mul_vector_number`(a JSON, n DOUBLE) RETURNS JSON DETERMINISTIC
BEGIN
    DECLARE i INT DEFAULT 0;
    DECLARE res JSON;
    SET res = JSON_ARRAY();
    WHILE i < JSON_LENGTH(a) DO
        SET res = JSON_ARRAYAPPEND(res, '$',
            CAST(JSON_EXTRACT(a, CONCAT('$[', i, ']')) AS DOUBLE) * n);
        SET i = i + 1;
    END WHILE;
    RETURN res;
END;
""".strip()


class MySqlInterpreter(ProceduralSqlInterpreter):

    # MySQL does not provide tanh/log1p natively: those are computed via
    # the AST-level fallback expressions.
    abs_function_name = "ABS"
    atan_function_name = "ATAN"
    exponent_function_name = "EXP"
    logarithm_function_name = "LN"
    log1p_function_name = NotImplemented
    power_function_name = "POW"
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
            MySqlCodeGenerator(indent=indent),
            indent=indent, function_name=function_name,
            feature_names=feature_names)

    def _linear_algebra_code(self):
        return LINEAR_ALGEBRA_CODE
