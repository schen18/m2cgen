import json
import os
from urllib.parse import urlparse

import pytest

from m2cgen.assemblers import get_assembler_cls
from m2cgen.interpreters import SqlInterpreter

from tests.e2e.executors.base import BaseExecutor


class SqlDuckdbExecutor(BaseExecutor):
    """Evaluates generated DuckDB macros in-process (no external services
    needed)."""

    def __init__(self, model):
        self.model = model
        self.interpreter = SqlInterpreter(dialect="duckdb")
        self.model_ast = get_assembler_cls(model)(model).assemble()
        self._con = None

    def prepare(self):
        import duckdb

        self._con = duckdb.connect()
        self._con.execute(self.interpreter.interpret(self.model_ast))
        self._n_features = self.interpreter.n_features

    def predict(self, X):
        placeholders = ", ".join(["?"] * self._n_features)
        result = self._con.execute(
            f"SELECT score({placeholders})",
            [float(v) for v in X[:self._n_features]]).fetchone()[0]
        if isinstance(result, list):
            return result
        return [result]


class _EnvGatedSqlExecutor(BaseExecutor):
    """Base for the MySQL/PostgreSQL e2e executors: they only run when a
    connection URI is provided via an environment variable and the
    corresponding driver is installed, otherwise the tests are skipped."""

    env_var = NotImplemented
    dialect = NotImplemented

    def __init__(self, model):
        uri = os.environ.get(self.env_var)
        if not uri:
            pytest.skip(f"{self.env_var} is not set")
        self._uri = urlparse(uri)
        self.model = model
        self.interpreter = SqlInterpreter(dialect=self.dialect)
        self.model_ast = get_assembler_cls(model)(model).assemble()
        self._conn = None

    def _connect(self):
        raise NotImplementedError

    def prepare(self):
        self._conn = self._connect()
        self._n_features = self.interpreter.n_features
        with self._conn.cursor() as cursor:
            cursor.execute(self.interpreter.interpret(self.model_ast))

    def _query(self, x):
        raise NotImplementedError

    def predict(self, X):
        result = self._query(X)
        if isinstance(result, (list, tuple)):
            return list(result)
        if isinstance(result, str):
            # MySQL JSON columns are returned as serialized strings
            return json.loads(result)
        return [result]


class MySqlExecutor(_EnvGatedSqlExecutor):

    env_var = "M2CGEN_MYSQL_URI"
    dialect = "mysql"

    def _connect(self):
        try:
            import pymysql
        except ImportError:
            pytest.skip("pymysql is not installed")
        return pymysql.connect(
            host=self._uri.hostname,
            port=self._uri.port or 3306,
            user=self._uri.username,
            password=self._uri.password,
            database=self._uri.path.lstrip("/"),
            charset="utf8mb4")

    def _query(self, x):
        placeholders = ", ".join(["%s"] * self._n_features)
        with self._conn.cursor() as cursor:
            cursor.execute(f"SELECT score({placeholders})",
                           tuple(float(v) for v in x[:self._n_features]))
            return cursor.fetchone()[0]


class PostgresExecutor(_EnvGatedSqlExecutor):

    env_var = "M2CGEN_POSTGRES_URI"
    dialect = "postgres"

    def _connect(self):
        try:
            import psycopg2
        except ImportError:
            pytest.skip("psycopg2 is not installed")
        return psycopg2.connect(self._uri.geturl())

    def _query(self, x):
        placeholders = ", ".join(["%s"] * self._n_features)
        with self._conn.cursor() as cursor:
            cursor.execute(f"SELECT score({placeholders})",
                           tuple(float(v) for v in x[:self._n_features]))
            return cursor.fetchone()[0]
