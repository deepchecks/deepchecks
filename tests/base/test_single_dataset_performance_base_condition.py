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
"""Test the conditions of the single dataset performance base check."""
import pandas as pd
from hamcrest import assert_that, contains_string, equal_to

from deepchecks.core import CheckResult, ConditionCategory
from deepchecks.core.check_utils.single_dataset_performance_base import BaseSingleDatasetPerformance


class DummySingleDatasetPerformance(BaseSingleDatasetPerformance):
    """Minimal concrete check, used to run the condition functions on a given result value."""

    def run(self, *args, **kwargs):
        """Not used - the conditions are evaluated on a pre-built result value."""


def run_condition(check, value):
    """Return the single condition result of the check for the given check value."""
    return check.conditions_decision(CheckResult(value))[0]


def test_condition_greater_than_fails_for_metric_with_missing_class():
    """A scalar metric is stored with a missing Class, and must still be evaluated."""
    # Arrange
    value = pd.DataFrame({'Metric': ['accuracy'], 'Class': [pd.NA], 'Value': [0.1]})
    check = DummySingleDatasetPerformance().add_condition_greater_than(0.9)

    # Act
    result = run_condition(check, value)

    # Assert
    assert_that(result.category, equal_to(ConditionCategory.FAIL))


def test_condition_greater_than_passes_for_metric_with_missing_class():
    """The same metric above the threshold still passes."""
    # Arrange
    value = pd.DataFrame({'Metric': ['accuracy'], 'Class': [pd.NA], 'Value': [0.95]})
    check = DummySingleDatasetPerformance().add_condition_greater_than(0.9)

    # Act
    result = run_condition(check, value)

    # Assert
    assert_that(result.category, equal_to(ConditionCategory.PASS))


def test_condition_greater_than_missing_class_alongside_real_classes():
    """Missing-class metrics are evaluated together with the per-class results."""
    # Arrange
    value = pd.DataFrame({
        'Metric': ['f1', 'f1', 'accuracy'],
        'Class': ['a', 'b', pd.NA],
        'Value': [0.95, 0.95, 0.1]
    })
    check = DummySingleDatasetPerformance().add_condition_greater_than(0.9)

    # Act
    result = run_condition(check, value)

    # Assert
    assert_that(result.category, equal_to(ConditionCategory.FAIL))


def test_condition_greater_than_class_mode_with_missing_class_present():
    """class_mode is still resolved per class when the metric also has a missing-class row."""
    # Arrange
    value = pd.DataFrame({
        'Metric': ['f1', 'f1', 'f1'],
        'Class': ['a', 'b', pd.NA],
        'Value': [0.95, 0.1, 0.99]
    })
    check = DummySingleDatasetPerformance().add_condition_greater_than(0.9, metrics=['f1'], class_mode='b')

    # Act
    result = run_condition(check, value)

    # Assert
    assert_that(result.category, equal_to(ConditionCategory.FAIL))


def test_condition_greater_than_unknown_class_mode():
    """An unknown class_mode is still reported as an error."""
    # Arrange
    value = pd.DataFrame({'Metric': ['f1'], 'Class': ['a'], 'Value': [0.95]})
    check = DummySingleDatasetPerformance().add_condition_greater_than(0.9, metrics=['f1'], class_mode='z')

    # Act
    result = run_condition(check, value)

    # Assert
    assert_that(result.category, equal_to(ConditionCategory.ERROR))
    assert_that(result.details, contains_string('class_mode expected be one of the classes'))
