#!/bin/bash

set -e

if [[ $TEST == "API" ]]; then
  make flake8 isort-check test-api run-codecov generate-code-examples
fi

if [[ $TEST == "E2E" ]]; then
  pip install --no-deps .
  rm -rfd m2cgen/
  EXTRA_ARGS=""
  # SQL macros of large models are slow to evaluate row-by-row; DuckDB is a
  # vectorized engine and shines on batched (table) execution instead.
  if [[ $LANG == "sql" ]]; then
    EXTRA_ARGS="--fast"
  fi
  pytest -v "-m=$LANG" $EXTRA_ARGS tests/e2e/
fi

if [[ $RELEASE == "true" ]]; then
  make package
fi
