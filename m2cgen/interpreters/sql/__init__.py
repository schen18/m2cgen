from m2cgen.interpreters.sql.base import resolve_feature_names
from m2cgen.interpreters.sql.duckdb import DuckDbInterpreter
from m2cgen.interpreters.sql.mysql import MySqlInterpreter
from m2cgen.interpreters.sql.postgres import PostgresInterpreter

DIALECT_TO_INTERPRETER = {
    "duckdb": DuckDbInterpreter,
    "mysql": MySqlInterpreter,
    "postgres": PostgresInterpreter,
}


class SqlInterpreter:
    """Facade which dispatches to a dialect-specific SQL interpreter.

    The default dialect is DuckDB. Feature references are rendered as
    function/macro parameters so that callers can pass their own columns
    without renaming: `SELECT score(col_a, col_b) FROM t`.
    """

    def __init__(self, dialect="duckdb", indent=4, function_name="score",
                 feature_names=None):
        try:
            interpreter_cls = DIALECT_TO_INTERPRETER[dialect]
        except KeyError:
            supported = ", ".join(sorted(DIALECT_TO_INTERPRETER))
            raise NotImplementedError(
                f"Unsupported SQL dialect '{dialect}'. "
                f"Supported dialects: {supported}")
        self._interpreter = interpreter_cls(
            indent=indent, function_name=function_name,
            feature_names=feature_names)

    def interpret(self, expr):
        code = self._interpreter.interpret(expr)
        # number of parameters of the generated function/macro (models may
        # use only a subset of the training features)
        self.n_features = self._interpreter._max_feature_index + 1
        return code


__all__ = [
    SqlInterpreter,
    DuckDbInterpreter,
    MySqlInterpreter,
    PostgresInterpreter,
    resolve_feature_names,
]
