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
"""Test functions of the train test label drift."""
from hamcrest import assert_that, close_to, equal_to, greater_than, has_entries, has_length

from deepchecks.core.condition import ConditionCategory
from deepchecks.tabular.checks import TrainTestLabelDrift
from tests.base.utils import equal_condition_result


def test_no_drift_classification_label(non_drifted_classification_label):
    # Arrange
    train, test = non_drifted_classification_label
    check = TrainTestLabelDrift(categorical_drift_method='PSI')

    # Act
    result = check.run(train, test)

    # Assert
    assert_that(result.value, has_entries(
        {'Drift score': close_to(0.003, 0.001),
         'Method': equal_to('PSI')}
    ))


def test_drift_classification_label(drifted_classification_label):
    # Arrange
    train, test = drifted_classification_label
    check = TrainTestLabelDrift(categorical_drift_method='PSI')

    # Act
    result = check.run(train, test)

    # Assert
    assert_that(result.value, has_entries(
        {'Drift score': close_to(0.24, 0.01),
         'Method': equal_to('PSI')}
    ))
    assert_that(result.display, has_length(greater_than(0)))


def test_drift_classification_label_without_display(drifted_classification_label):
    # Arrange
    train, test = drifted_classification_label
    check = TrainTestLabelDrift(categorical_drift_method='PSI')

    # Act
    result = check.run(train, test, with_display=False)

    # Assert
    assert_that(result.value, has_entries(
        {'Drift score': close_to(0.24, 0.01),
         'Method': equal_to('PSI')}
    ))
    assert_that(result.display, has_length(0))


def test_drift_regression_label(drifted_regression_label):
    # Arrange
    train, test = drifted_regression_label
    check = TrainTestLabelDrift(categorical_drift_method='PSI')

    # Act
    result = check.run(train, test)

    # Assert
    assert_that(result.value, has_entries(
        {'Drift score': close_to(0.34, 0.01),
         'Method': equal_to('Earth Mover\'s Distance')}
    ))


def test_reduce_output_drift_regression_label(drifted_regression_label):
    # Arrange
    train, test = drifted_regression_label
    check = TrainTestLabelDrift(categorical_drift_method='PSI')

    # Act
    result = check.run(train, test)

    # Assert
    assert_that(result.reduce_output(), has_entries(
        {'Label Drift Score': close_to(0.34, 0.01)}
    ))

    assert_that(check.greater_is_better(), equal_to(False))


def test_drift_max_drift_score_condition_fail_psi(drifted_classification_label):
    # Arrange
    train, test = drifted_classification_label
    check = TrainTestLabelDrift(categorical_drift_method='PSI').add_condition_drift_score_less_than()

    # Act
    result = check.run(train, test)
    condition_result, *_ = check.conditions_decision(result)

    # Assert
    assert_that(condition_result, equal_condition_result(
        is_pass=False,
        name='categorical drift score < 0.2 and numerical drift score < 0.1 for label drift',
        details='Label\'s drift score PSI is 0.24'
    ))


def test_drift_max_drift_score_condition_fail_emd(drifted_regression_label):
    # Arrange
    train, test = drifted_regression_label
    check = TrainTestLabelDrift(categorical_drift_method='PSI').add_condition_drift_score_less_than()

    # Act
    result = check.run(train, test)
    condition_result, *_ = check.conditions_decision(result)

    # Assert
    assert_that(condition_result, equal_condition_result(
        is_pass=False,
        category=ConditionCategory.FAIL,
        name='categorical drift score < 0.2 and numerical drift score < 0.1 for label drift',
        details='Label\'s drift score Earth Mover\'s Distance is 0.34'
    ))


def test_drift_max_drift_score_condition_pass_threshold(non_drifted_classification_label):
    # Arrange
    train, test = non_drifted_classification_label
    check = TrainTestLabelDrift(categorical_drift_method='PSI') \
        .add_condition_drift_score_less_than(max_allowed_categorical_score=1,
                                             max_allowed_numeric_score=1)

    # Act
    result = check.run(train, test)
    condition_result, *_ = check.conditions_decision(result)

    # Assert
    assert_that(condition_result, equal_condition_result(
        is_pass=True,
        details='Label\'s drift score PSI is 3.37E-3',
        name='categorical drift score < 1 and numerical drift score < 1 for label drift'
    ))
