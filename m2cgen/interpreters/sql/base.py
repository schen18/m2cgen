"""Shared plumbing for the SQL interpreters: feature-name resolution,
identifier sanitization and feature-reference rendering.

The SQL interpreters differ from all other interpreters in one fundamental
way: instead of indexing into an input array, feature references are
rendered as identifiers (function parameters / macro parameters), so that
callers can pass their table's own columns without renaming anything.
"""
import re

# SQL reserved words which must not be used as unquoted identifiers
# (DuckDB macro parameters cannot be quoted, so sanitization has to
# guarantee that the generated names are safe).
RESERVED_WORDS = {
    "all", "and", "any", "array", "as", "asc", "asymmetric", "both",
    "case", "cast", "check", "collate", "column", "constraint", "create",
    "cross", "current_date", "current_time", "current_timestamp",
    "current_user", "default", "deferrable", "desc", "distinct", "else",
    "end", "except", "exists", "false", "for", "foreign", "from", "full",
    "grant", "group", "having", "in", "inner", "intersect", "into", "is",
    "join", "lateral", "leading", "left", "like", "limit", "not", "null",
    "offset", "on", "or", "order", "outer", "over", "placing", "primary",
    "references", "right", "select", "some", "symmetric", "table", "then",
    "trailing", "true", "union", "unique", "user", "using", "when", "where",
    "window", "with",
}


def sanitize_identifier(name, fallback):
    """Turns an arbitrary feature name into a safe SQL identifier."""
    result = re.sub(r"[^0-9A-Za-z_]", "_", str(name))
    if not result or result[0].isdigit():
        result = "f_" + result
    if result.lower() in RESERVED_WORDS:
        result += "_"
    return result or fallback


def resolve_feature_names(model):
    """Extracts feature names from a fitted model, when it recorded them.

    Returns a list of raw names, or None if the model does not carry
    feature names (e.g. it was trained on plain numpy arrays).
    """
    # scikit-learn estimators fitted on pandas DataFrames
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        return [str(n) for n in names]

    # LightGBM (sklearn API)
    booster = getattr(model, "booster_", None)
    if booster is not None and hasattr(booster, "feature_name"):
        names = booster.feature_name()
        if names:
            return [str(n) for n in names]

    # XGBoost (sklearn API)
    get_booster = getattr(model, "get_booster", None)
    if get_booster is not None:
        names = get_booster().feature_names
        if names:
            return [str(n) for n in names]

    # statsmodels results wrappers: exog names without the constant column
    data = getattr(getattr(model, "model", None), "data", None)
    exog_names = getattr(data, "exog_names", None)
    if exog_names:
        const_idx = getattr(data, "const_idx", None)
        return [str(n) for idx, n in enumerate(exog_names) if idx != const_idx]

    return None


class SqlFeatureRefMixin:
    """Renders FeatureRef expressions as identifiers (one per feature)
    instead of array accesses, and tracks how many features are used.

    Subclasses decide how identifiers are rendered (quoted or not) by
    implementing `_render_feature_ident(index, name)`.
    """

    def __init__(self, *args, feature_names=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._raw_feature_names = feature_names
        self._feature_idents = {}
        self._max_feature_index = -1

    def _feature_name(self, index):
        if (self._raw_feature_names is not None
                and index < len(self._raw_feature_names)):
            raw = self._raw_feature_names[index]
        else:
            raw = None
        return raw if raw not in (None, "") else f"feature_{index}"

    def _get_feature_ident(self, index):
        if index not in self._feature_idents:
            self._feature_idents[index] = self._render_feature_ident(
                index, sanitize_identifier(self._feature_name(index),
                                           f"feature_{index}"))
        return self._feature_idents[index]

    def _render_feature_ident(self, index, name):
        raise NotImplementedError

    def interpret_feature_ref(self, expr, **kwargs):
        self._max_feature_index = max(self._max_feature_index, expr.index)
        return self._get_feature_ident(expr.index)

    def _reset_sql_state(self):
        self._feature_idents = {}
        self._max_feature_index = -1
