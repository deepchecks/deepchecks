# ----------------------------------------------------------------------------
# Copyright (C) 2021 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""Test functions of the train test label drift."""
from hamcrest import assert_that, has_entries, close_to, equal_to

from deepchecks.checks import TrainTestLabelDrift
from tests.checks.utils import equal_condition_result


def test_no_drift_classification_label(non_drifted_classification_label):
    # Arrange
    train, test = non_drifted_classification_label
    check = TrainTestLabelDrift()

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
    check = TrainTestLabelDrift()

    # Act
    result = check.run(train, test)

    # Assert
    assert_that(result.value, has_entries(
            {'Drift score': close_to(0.24, 0.01),
             'Method': equal_to('PSI')}
    ))


def test_drift_regression_label(drifted_regression_label):
    # Arrange
    train, test = drifted_regression_label
    check = TrainTestLabelDrift()

    # Act
    result = check.run(train, test)

    # Assert
    assert_that(result.value, has_entries(
        {'Drift score': close_to(0.249, 0.01),
         'Method': equal_to('Earth Mover\'s Distance')}
         ))


def test_drift_max_drift_score_condition_fail_psi(drifted_classification_label):
    # Arrange
    train, test = drifted_classification_label
    check = TrainTestLabelDrift().add_condition_drift_score_not_greater_than()

    # Act
    result = check.run(train, test)
    condition_result, *_ = check.conditions_decision(result)

    # Assert
    assert_that(condition_result, equal_condition_result(
        is_pass=False,
        name='PSI <= 0.2 and Earth Mover\'s Distance <= 0.1 for label drift',
        details='Found label PSI above threshold: 0.24'
    ))


def test_drift_max_drift_score_condition_fail_emd(drifted_regression_label):
    # Arrange
    train, test = drifted_regression_label
    check = TrainTestLabelDrift().add_condition_drift_score_not_greater_than()

    # Act
    result = check.run(train, test)
    condition_result, *_ = check.conditions_decision(result)

    # Assert
    assert_that(condition_result, equal_condition_result(
        is_pass=False,
        name='PSI <= 0.2 and Earth Mover\'s Distance <= 0.1 for label drift',
        details='Label\'s Earth Mover\'s Distance above threshold: 0.26'
    ))


def test_drift_max_drift_score_condition_pass_threshold(non_drifted_classification_label):
    # Arrange
    train, test = non_drifted_classification_label
    check = TrainTestLabelDrift().add_condition_drift_score_not_greater_than(max_allowed_psi_score=1,
                                                                             max_allowed_earth_movers_score=1)

    # Act
    result = check.run(train, test)
    condition_result, *_ = check.conditions_decision(result)

    # Assert
    assert_that(condition_result, equal_condition_result(
        is_pass=True,
        name='PSI <= 1 and Earth Mover\'s Distance <= 1 for label drift'
    ))
