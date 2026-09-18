import pytest

from m2cgen import ast
from m2cgen.interpreters import SqlInterpreter

from tests.utils import assert_code_equal


def test_duckdb_if_expr():
    expr = ast.IfExpr(
        ast.CompExpr(ast.FeatureRef(0), ast.NumVal(1.5), ast.CompOpType.LT),
        ast.NumVal(2),
        ast.NumVal(3))

    expected_code = """
CREATE OR REPLACE MACRO score(feature_0) AS (
    SELECT CASE WHEN feature_0 < 1.5E0 THEN 2.0E0 ELSE 3.0E0 END
);
"""

    interpreter = SqlInterpreter(dialect="duckdb")
    assert_code_equal(interpreter.interpret(expr), expected_code)


def test_duckdb_bin_num_expr():
    expr = ast.BinNumExpr(
        ast.BinNumExpr(
            ast.FeatureRef(0), ast.NumVal(-2), ast.BinNumOpType.DIV),
        ast.NumVal(2),
        ast.BinNumOpType.MUL)

    expected_code = """
CREATE OR REPLACE MACRO score(feature_0) AS (
    SELECT feature_0 / -2.0E0 * 2.0E0
);
"""

    interpreter = SqlInterpreter(dialect="duckdb")
    assert_code_equal(interpreter.interpret(expr), expected_code)


def test_duckdb_eq_comp():
    expr = ast.IfExpr(
        ast.CompExpr(ast.NumVal(1), ast.FeatureRef(0), ast.CompOpType.EQ),
        ast.NumVal(2),
        ast.NumVal(3))

    expected_code = """
CREATE OR REPLACE MACRO score(feature_0) AS (
    SELECT CASE WHEN 1.0E0 = feature_0 THEN 2.0E0 ELSE 3.0E0 END
);
"""

    interpreter = SqlInterpreter(dialect="duckdb")
    assert_code_equal(interpreter.interpret(expr), expected_code)


def test_duckdb_feature_names_and_tanh():
    expr = ast.TanhExpr(ast.FeatureRef(0))

    expected_code = """
CREATE OR REPLACE MACRO score(age) AS (
    SELECT tanh(age)
);
"""

    interpreter = SqlInterpreter(dialect="duckdb", feature_names=["age"])
    assert_code_equal(interpreter.interpret(expr), expected_code)


def test_duckdb_reused_expr_linearization():
    pow_expr = ast.PowExpr(ast.FeatureRef(0), ast.NumVal(2), to_reuse=True)
    expr = ast.BinNumExpr(pow_expr, pow_expr, ast.BinNumOpType.ADD)

    expected_code = """
CREATE OR REPLACE MACRO score(feature_0) AS (
    WITH
        v0 AS (SELECT power(feature_0, 2.0E0) AS x)
    SELECT (SELECT x FROM v0) + (SELECT x FROM v0)
);
"""

    interpreter = SqlInterpreter(dialect="duckdb")
    assert_code_equal(interpreter.interpret(expr), expected_code)


def test_duckdb_vector_output():
    expr = ast.BinVectorNumExpr(
        ast.VectorVal([ast.NumVal(1), ast.NumVal(2)]),
        ast.NumVal(0.5),
        ast.BinNumOpType.MUL)

    expected_code = """
CREATE OR REPLACE MACRO score() AS (
    SELECT list_transform([1.0E0, 2.0E0], x -> x * 0.5E0)
);
"""

    interpreter = SqlInterpreter(dialect="duckdb")
    assert_code_equal(interpreter.interpret(expr), expected_code)


def test_mysql_if_expr():
    expr = ast.IfExpr(
        ast.CompExpr(ast.FeatureRef(0), ast.NumVal(1.5), ast.CompOpType.LT),
        ast.NumVal(2),
        ast.NumVal(3))

    expected_code = """
CREATE FUNCTION `score`(`feature_0` DOUBLE) RETURNS DOUBLE DETERMINISTIC
BEGIN
    DECLARE var0 DOUBLE;

    IF `feature_0` < 1.5 THEN
        SET var0 = 2.0;
    ELSE
        SET var0 = 3.0;
    END IF;
    RETURN var0;
END
"""

    interpreter = SqlInterpreter(dialect="mysql")
    assert_code_equal(interpreter.interpret(expr), expected_code)


def test_mysql_tanh_fallback():
    expr = ast.TanhExpr(ast.NumVal(2))

    code = SqlInterpreter(dialect="mysql").interpret(expr)

    # MySQL has no TANH: the exp-based AST fallback must be used
    assert "EXP(" in code
    assert "TANH" not in code


def test_mysql_feature_names_are_quoted():
    expr = ast.FeatureRef(0)

    code = SqlInterpreter(
        dialect="mysql", feature_names=["weird name!"]).interpret(expr)

    assert "`weird_name_` DOUBLE" in code


def test_postgres_if_expr():
    expr = ast.IfExpr(
        ast.CompExpr(ast.FeatureRef(0), ast.NumVal(1.5), ast.CompOpType.LT),
        ast.NumVal(2),
        ast.NumVal(3))

    expected_code = """
CREATE FUNCTION "score"("feature_0" double precision) RETURNS double precision
LANGUAGE plpgsql IMMUTABLE AS $$
DECLARE
    var0 double precision;
BEGIN
    IF "feature_0" < 1.5 THEN
        var0 := 2.0;
    ELSE
        var0 := 3.0;
    END IF;
    RETURN var0;
END
$$;
"""

    interpreter = SqlInterpreter(dialect="postgres")
    assert_code_equal(interpreter.interpret(expr), expected_code)


def test_postgres_vector_output_with_linear_algebra():
    expr = ast.BinVectorNumExpr(
        ast.VectorVal([ast.NumVal(1), ast.NumVal(2)]),
        ast.NumVal(0.5),
        ast.BinNumOpType.MUL)

    code = SqlInterpreter(dialect="postgres").interpret(expr)

    assert 'RETURN m2c_mul_vector_number(ARRAY[1.0, 2.0], 0.5);' in code
    assert "CREATE FUNCTION m2c_mul_vector_number" in code


def test_reserved_word_feature_name():
    expr = ast.FeatureRef(0)

    code = SqlInterpreter(dialect="duckdb", feature_names=["order"]).interpret(
        expr)

    assert "MACRO score(order_)" in code


def test_unknown_dialect():
    with pytest.raises(NotImplementedError, match="Unsupported SQL dialect"):
        SqlInterpreter(dialect="oracle")
