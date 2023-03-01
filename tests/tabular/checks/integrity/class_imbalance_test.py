# ----------------------------------------------------------------------------
# Copyright (C) 2021-2023 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""Contains unit tests for the class_imbalance check."""
import pandas as pd
from hamcrest import assert_that, calling, equal_to, has_items, has_length, raises

from deepchecks.core import ConditionCategory
from deepchecks.core.errors import DeepchecksValueError, ModelValidationError
from deepchecks.tabular.checks.data_integrity.class_imbalance import ClassImbalance
from deepchecks.tabular.dataset import Dataset
from tests.base.utils import equal_condition_result


def test_dataset_wrong_input():
    """DeepchecksValueError is raised when str is passed as Dataset"""
    x = 'wrong_input'
    assert_that(
        calling(ClassImbalance().run).with_args(x),
        raises(
            DeepchecksValueError,
            'non-empty instance of Dataset or DataFrame was expected, instead got str'
        )
    )


def test_label_imbalance_condition_pass():

    # Arrange
    data = {
        'col': [1, 1, 1, 2, 2, 2] * 100,
        'label': [1, 1, 1, 1, 2, 1] * 100
    }
    ds = Dataset(pd.DataFrame(data), label='label')
    check = ClassImbalance().add_condition_class_ratio_less_than(.3)

    # Act
    result = check.run(ds)
    condition_result = check.conditions_decision(result)

    # Assert
    assert_that(condition_result, has_items(
        equal_condition_result(is_pass=True,
                               details='The ratio between least to most frequent label is 0.2',
                               name='The ratio between least frequent label to most frequent label '
                                    'is less than or equal 0.3')
    ))


def test_label_imbalance_condition_warn():

    data = {
        'col': [1, 2, 3, 4] * 100,
        'label': ['a', 'a', 'b', 'c'] * 100
    }
    ds = Dataset(pd.DataFrame(data), label='label')
    check = ClassImbalance().add_condition_class_ratio_less_than(.1)

    result = check.run(ds)
    condition_result = check.conditions_decision(result)

    assert_that(condition_result, has_items(
        equal_condition_result(is_pass=False,
                               details='The ratio between least to most frequent label is 0.5',
                               name='The ratio between least frequent label to most frequent label '
                                    'is less than or equal 0.1',
                               category=ConditionCategory.WARN)
    ))


def test_result():
    """validate CheckResult output"""
    data = {
        'col': [1, 2, 3, 4, 5, 6] * 100,
        'label': ['a', 'a', 'a', 'b', 'c', 'c'] * 100
    }
    ds = Dataset(pd.DataFrame(data), label='label')
    res = ClassImbalance().run(ds)

    expected_res_dict = {'a': 0.5, 'c': 0.33, 'b': 0.17}
    assert_that(res.value, equal_to(expected_res_dict))


def test_condition_input_validation():
    """ModelValidationError is raised for regressions tasks"""
    data = {
        'col': [1, 2, 3, 4, 5, 6] * 100,
        'label': list(range(6)) * 100
    }
    ds = Dataset(pd.DataFrame(data), label='label', label_type='regression')
    assert_that(
        calling(ClassImbalance().run).with_args(ds),
        raises(ModelValidationError, 'Check is irrelevant for regression tasks'))


def test_ignore_nan_false():
    """check result with None in class"""

    data = {
        'col': [1, 2, 3, 4, 5, 6] * 100,
        'label': ['a', 'a', 'a', None, 'b', 'b'] * 100
    }
    ds = Dataset(pd.DataFrame(data), label='label')

    result = ClassImbalance(ignore_nan=False).run(ds)
    assert_that(result.value, has_length(3))


def test_ignore_nan_true():
    """check result with None in class"""

    data = {
        'col': [1, 2, 3, 4, 5, 6] * 100,
        'label': ['a', 'a', 'a', None, 'b', 'b'] * 100
    }
    ds = Dataset(pd.DataFrame(data), label='label')

    # Act
    result = ClassImbalance(ignore_nan=True).run(ds)
    expected_result_value = {'a': 0.6, 'b': 0.4}

    # Assert
    assert_that(result.value, equal_to(expected_result_value))
