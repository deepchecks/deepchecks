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
from hamcrest import assert_that, close_to, equal_to, greater_than, has_entries, has_length

from deepchecks.tabular.checks import TrainTestPredictionDrift
from tests.base.utils import equal_condition_result


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

def test_reduce_no_drift_classification_label(diabetes, diabetes_model):
    # Arrange
    train, test = diabetes
    check = TrainTestPredictionDrift(categorical_drift_method='PSI')

    # Act
    result = check.run(train, test, diabetes_model)

    # Assert
    assert_that(result.reduce_output(), has_entries(
            {'Prediction Drift Score': close_to(0.04, 0.01)}
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
    assert_that(result.display, has_length(greater_than(0)))


def test_drift_classification_label_without_display(drifted_data_and_model):
    # Arrange
    train, test, model = drifted_data_and_model
    check = TrainTestPredictionDrift(categorical_drift_method='PSI')

    # Act
    result = check.run(train, test, model, with_display=False)

    # Assert
    assert_that(result.value, has_entries(
            {'Drift score': close_to(0.78, 0.01),
             'Method': equal_to('PSI')}
    ))
    assert_that(result.display, has_length(0))


def test_drift_classification_label_cramer(drifted_data_and_model):
    # Arrange
    train, test, model = drifted_data_and_model
    check = TrainTestPredictionDrift(categorical_drift_method='cramer_v')

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
    check = TrainTestPredictionDrift(categorical_drift_method='PSI').add_condition_drift_score_less_than()

    # Act
    result = check.run(train, test, model)
    condition_result, *_ = check.conditions_decision(result)

    # Assert
    assert_that(condition_result, equal_condition_result(
        is_pass=False,
        name='categorical drift score < 0.15 and numerical drift score < 0.075',
        details='Found model prediction PSI drift score of 0.79'
    ))


def test_drift_max_drift_score_condition_pass_threshold(drifted_data_and_model):
    # Arrange
    train, test, model = drifted_data_and_model
    check = TrainTestPredictionDrift(categorical_drift_method='PSI') \
        .add_condition_drift_score_less_than(max_allowed_categorical_score=1,
                                             max_allowed_numeric_score=1)

    # Act
    result = check.run(train, test, model)
    condition_result, *_ = check.conditions_decision(result)

    # Assert
    assert_that(condition_result, equal_condition_result(
        is_pass=True,
        details='Found model prediction PSI drift score of 0.79',
        name='categorical drift score < 1 and numerical drift score < 1'
    ))
