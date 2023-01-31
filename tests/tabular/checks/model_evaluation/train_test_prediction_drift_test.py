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
from hamcrest import assert_that, calling, close_to, equal_to, greater_than, has_entries, has_length, raises

from deepchecks.core.errors import DeepchecksValueError
from deepchecks.tabular.checks import TrainTestPredictionDrift
from tests.base.utils import equal_condition_result


def test_no_drift_regression_label(diabetes, diabetes_model):
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


def test_reduce_no_drift_regression_label(diabetes, diabetes_model):
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
    check = TrainTestPredictionDrift(categorical_drift_method='PSI', drift_mode='prediction')

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
    check = TrainTestPredictionDrift(categorical_drift_method='PSI', drift_mode='prediction')

    # Act
    result = check.run(train, test, model, with_display=False)

    # Assert
    assert_that(result.value, has_entries(
        {'Drift score': close_to(0.78, 0.01),
         'Method': equal_to('PSI')}
    ))
    assert_that(result.display, has_length(0))


def test_drift_regression_label_raise_on_proba(diabetes, diabetes_model):
    # Arrange
    train, test = diabetes
    check = TrainTestPredictionDrift(categorical_drift_method='PSI', drift_mode='proba')

    # Act & Assert
    assert_that(calling(check.run).with_args(train, test, diabetes_model),
                raises(DeepchecksValueError,
                       'probability_drift="proba" is not supported for regression tasks'))


def test_drift_regression_label_cramer(drifted_data_and_model):
    # Arrange
    train, test, model = drifted_data_and_model
    check = TrainTestPredictionDrift(categorical_drift_method='cramer_v', drift_mode='prediction')

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
    check = TrainTestPredictionDrift(categorical_drift_method='PSI', drift_mode='prediction'
                                     ).add_condition_drift_score_less_than()

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
    check = TrainTestPredictionDrift(categorical_drift_method='PSI', drift_mode='prediction') \
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


def test_multiclass_proba(iris_split_dataset_and_model_rf):
    # Arrange
    train, test, model = iris_split_dataset_and_model_rf
    check = TrainTestPredictionDrift(categorical_drift_method='PSI', max_num_categories=10, min_category_size_ratio=0,
                                     drift_mode='proba')

    # Act
    result = check.run(train, test, model)

    # Assert
    assert_that(result.value, has_entries(
        {'Drift score': has_entries({0: close_to(0.06, 0.01), 1: close_to(0.06, 0.01), 2: close_to(0.03, 0.01)}),
         'Method': equal_to('Earth Mover\'s Distance')}
    ))

    assert_that(result.display, has_length(5))


def test_binary_proba_condition_fail_threshold(drifted_data_and_model):
    # Arrange
    train, test, model = drifted_data_and_model
    check = TrainTestPredictionDrift(categorical_drift_method='PSI', drift_mode='proba'
                                     ).add_condition_drift_score_less_than()

    # Act
    result = check.run(train, test, model)
    condition_result, *_ = check.conditions_decision(result)

    # Assert
    assert_that(result.value, has_entries(
        {'Drift score': close_to(0.23, 0.01),
         'Method': equal_to('Earth Mover\'s Distance')}
    ))

    assert_that(condition_result, equal_condition_result(
        is_pass=False,
        name='categorical drift score < 0.15 and numerical drift score < 0.075',
        details='Found model prediction Earth Mover\'s Distance drift score of 0.23'
    ))


def test_multiclass_proba_reduce_aggregations(iris_split_dataset_and_model_rf):
    # Arrange
    train, test, model = iris_split_dataset_and_model_rf
    check = TrainTestPredictionDrift(categorical_drift_method='PSI', max_num_categories=10, min_category_size_ratio=0,
                                     drift_mode='proba', aggregation_method='weighted'
                                     ).add_condition_drift_score_less_than(max_allowed_numeric_score=0.05)

    # Act
    result = check.run(train, test, model)

    # Assert
    assert_that(result.reduce_output(), has_entries(
        {'Weighted Drift Score': close_to(0.05, 0.01)}
    ))

    check.aggregation_method = 'mean'
    assert_that(result.reduce_output(), has_entries(
        {'Mean Drift Score': close_to(0.05, 0.01)}
    ))

    check.aggregation_method = 'max'
    assert_that(result.reduce_output(), has_entries(
        {'Max Drift Score': close_to(0.06, 0.01)}
    ))

    check.aggregation_method = 'none'
    assert_that(result.reduce_output(), has_entries(
        {'Drift Score class 0': close_to(0.06, 0.01), 'Drift Score class 1': close_to(0.06, 0.01),
         'Drift Score class 2': close_to(0.03, 0.01)})
    )

    # Test condition
    condition_result, *_ = check.conditions_decision(result)

    assert_that(condition_result, equal_condition_result(
        is_pass=False,
        name='categorical drift score < 0.15 and numerical drift score < 0.05',
        details='Found 2 classes with model predicted probability Earth Mover\'s '
                'Distance drift score above threshold: 0.05.'
    ))
