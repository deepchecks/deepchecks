# ----------------------------------------------------------------------------
# Copyright (C) 2021-2022 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""Test functions of the train test label drift."""
from hamcrest import assert_that, close_to, equal_to, has_entries

from deepchecks.tabular.checks import TrainTestPredictionDrift
from tests.checks.utils import equal_condition_result


def test_no_drift_classification_label(diabetes, diabetes_model):
    # Arrange
    train, test = diabetes
    check = TrainTestPredictionDrift(categorical_drift_method='PSI')

    # Act
    result = check.run(train, test, diabetes_model)

    # Assert
    assert_that(result.value, has_entries(
            {'Drift score': close_to(0.04, 0.01),
             'Method': equal_to('Earth Mover\'s Distance')}
    ))


def test_drift_classification_label(drifted_data_and_model):
    # Arrange
    train, test, model = drifted_data_and_model
    check = TrainTestPredictionDrift(categorical_drift_method='PSI')

    # Act
    result = check.run(train, test, model)

    # Assert
    assert_that(result.value, has_entries(
            {'Drift score': close_to(0.78, 0.01),
             'Method': equal_to('PSI')}
    ))


def test_drift_classification_label_cramer(drifted_data_and_model):
    # Arrange
    train, test, model = drifted_data_and_model
    check = TrainTestPredictionDrift(categorical_drift_method='Cramer')

    # Act
    result = check.run(train, test, model)

    # Assert
    assert_that(result.value, has_entries(
            {'Drift score': close_to(0.426, 0.01),
             'Method': equal_to('Cramer\'s V')}
    ))


def test_drift_max_drift_score_condition_fail_psi(drifted_data_and_model):
    # Arrange
    train, test, model = drifted_data_and_model
    check = TrainTestPredictionDrift(categorical_drift_method='PSI').add_condition_drift_score_not_greater_than()

    # Act
    result = check.run(train, test, model)
    condition_result, *_ = check.conditions_decision(result)

    # Assert
    assert_that(condition_result, equal_condition_result(
        is_pass=False,
        name='categorical drift score <= 0.15 and numerical drift score <= 0.075',
        details='Found model prediction PSI above threshold: 0.79'
    ))


def test_drift_max_drift_score_condition_pass_threshold(drifted_data_and_model):
    # Arrange
    train, test, model = drifted_data_and_model
    check = TrainTestPredictionDrift(categorical_drift_method='PSI') \
        .add_condition_drift_score_not_greater_than(max_allowed_categorical_score=1,
                                                    max_allowed_numerical_score=1)

    # Act
    result = check.run(train, test, model)
    condition_result, *_ = check.conditions_decision(result)

    # Assert
    assert_that(condition_result, equal_condition_result(
        is_pass=True,
        name='categorical drift score <= 1 and numerical drift score <= 1'
    ))
