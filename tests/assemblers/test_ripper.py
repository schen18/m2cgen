import pytest
import wittgenstein as lw

from m2cgen import assemblers


def test_ripper():
    estimator = lw.RIPPER(random_state=1)
    from tests import utils
    utils.assert_model_predictions_match(
        utils.get_binary_classification_model_trainer(), estimator)


def test_irep():
    estimator = lw.IREP(random_state=1)
    from tests import utils
    utils.assert_model_predictions_match(
        utils.get_binary_classification_model_trainer(), estimator)


def test_non_numeric_condition_not_supported():
    class _Cond:
        def __init__(self, feature, val):
            self.feature = feature
            self.val = val

    with pytest.raises(NotImplementedError, match="numeric"):
        assemblers.WittgensteinRuleModelAssembler._cond_comparisons(
            _Cond("f0", "male"), {"f0": 0})
