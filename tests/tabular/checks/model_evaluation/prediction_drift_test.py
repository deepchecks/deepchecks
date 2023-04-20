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
"""Test functions of the label drift."""
import pandas as pd
from hamcrest import assert_that, calling, close_to, equal_to, greater_than, has_entries, has_length, raises

from deepchecks.core.errors import DeepchecksValueError, NotEnoughSamplesError
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import PredictionDrift
from tests.base.utils import equal_condition_result


def remove_label(ds: Dataset) -> Dataset:
    """Remove the label from the dataset."""
    return Dataset(ds.data.drop(columns=ds.label_name, axis=1), cat_features=ds.cat_features)


def test_no_drift_regression_label_emd(diabetes, diabetes_model):
    # Arrange
    train, test = diabetes
    # Remove labels
    train = remove_label(train)
    test = remove_label(test)
    check = PredictionDrift(categorical_drift_method='PSI', numerical_drift_method='EMD')

    # Act
    result = check.run(train, test, diabetes_model)

    # Assert
    assert_that(result.value, has_entries(
        {'Drift score': close_to(0.04, 0.01),
         'Method': equal_to('Earth Mover\'s Distance')}
    ))


def test_no_drift_regression_label_ks(diabetes, diabetes_model):
    # Arrange
    train, test = diabetes
    check = PredictionDrift(numerical_drift_method='KS')

    # Act
    result = check.run(train, test, diabetes_model)

    # Assert
    assert_that(result.value, has_entries(
        {'Drift score': close_to(0.11, 0.01),
         'Method': equal_to('Kolmogorov-Smirnov')}
    ))


def test_reduce_no_drift_regression_label(diabetes, diabetes_model):
    # Arrange
    train, test = diabetes
    train = remove_label(train)
    test = remove_label(test)
    check = PredictionDrift(categorical_drift_method='PSI', numerical_drift_method='EMD')

    # Act
    result = check.run(train, test, diabetes_model)

    # Assert
    assert_that(result.reduce_output(), has_entries(
        {'Prediction Drift Score': close_to(0.04, 0.01)}
    ))


def test_drift_classification_label(drifted_data_and_model):
    # Arrange
    train, test, model = drifted_data_and_model
    train = remove_label(train)
    test = remove_label(test)
    check = PredictionDrift(categorical_drift_method='PSI', drift_mode='prediction')

    # Act
    result = check.run(train, test, model)

    # Assert
    assert_that(result.value, has_entries(
        {'Drift score': close_to(0.78, 0.01),
         'Method': equal_to('PSI')}
    ))
    assert_that(result.display, has_length(greater_than(0)))


def test_drift_not_enough_samples(drifted_data_and_model):
    # Arrange
    train, test, model = drifted_data_and_model
    train = remove_label(train)
    test = remove_label(test)
    check = PredictionDrift(min_samples=1000000)

    # Assert
    assert_that(calling(check.run).with_args(train, test, model),
                raises(NotEnoughSamplesError))


def test_drift_classification_label_without_display(drifted_data_and_model):
    # Arrange
    train, test, model = drifted_data_and_model
    check = PredictionDrift(categorical_drift_method='PSI', drift_mode='prediction')

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
    train = remove_label(train)
    test = remove_label(test)
    check = PredictionDrift(categorical_drift_method='PSI', drift_mode='proba')

    # Act & Assert
    assert_that(calling(check.run).with_args(train, test, diabetes_model),
                raises(DeepchecksValueError,
                       'probability_drift="proba" is not supported for regression tasks'))


def test_drift_regression_label_cramer(drifted_data_and_model):
    # Arrange
    train, test, model = drifted_data_and_model
    check = PredictionDrift(categorical_drift_method='cramers_v', drift_mode='prediction')

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
    train = remove_label(train)
    test = remove_label(test)
    check = PredictionDrift(categorical_drift_method='PSI', drift_mode='prediction'
                                     ).add_condition_drift_score_less_than()

    # Act
    result = check.run(train, test, model)
    condition_result, *_ = check.conditions_decision(result)

    # Assert
    assert_that(condition_result, equal_condition_result(
        is_pass=False,
        name='Prediction drift score < 0.15',
        details='Found model prediction PSI drift score of 0.79'
    ))


def test_balance_classes_without_cramers_v(drifted_data_and_model):
    # Arrange
    train, test, model = drifted_data_and_model
    train = remove_label(train)
    test = remove_label(test)
    check = PredictionDrift(categorical_drift_method='PSI', drift_mode='prediction', balance_classes=True)

    assert_that(calling(check.run).with_args(train, test, model),
                raises(DeepchecksValueError,
                       'balance_classes is only supported for Cramer\'s V. please set balance_classes=False '
                       'or use \'cramers_v\' as categorical_drift_method'))


def test_balance_classes_without_correct_drift_mode():
    # Arrange
    assert_that(calling(PredictionDrift).with_args(balance_classes=True, drift_mode='proba'),
                raises(DeepchecksValueError,
                       'balance_classes=True is not supported for drift_mode=\'proba\'. '
                       'Change drift_mode to \'prediction\' or \'auto\' in order to use this parameter'))

def test_balance_classes_with_drift_mode_auto(drifted_data):
    # Arrange
    train, test = drifted_data
    train = remove_label(train)
    test = remove_label(test)

    n_train = train.n_samples
    n_test = test.n_samples

    predictions_train = [0] * int(n_train * 0.95) + [1] * int(n_train * 0.05)
    predictions_test = [0] * int(n_test * 0.96) + [1] * int(n_test * 0.04)
    check = PredictionDrift(balance_classes=True)

    # Act
    result = check.run(train, test, y_pred_train=predictions_train, y_pred_test=predictions_test)

    # Assert
    assert_that(result.value, has_entries(
        {'Drift score': close_to(0.05, 0.01),
         'Method': equal_to('Cramer\'s V')} # If cramer's V then proves it changed to prediction mode
    ))


def test_drift_max_drift_score_condition_pass_threshold(drifted_data_and_model):
    # Arrange
    train, test, model = drifted_data_and_model
    train = remove_label(train)
    test = remove_label(test)
    check = PredictionDrift(categorical_drift_method='PSI', drift_mode='prediction') \
        .add_condition_drift_score_less_than(max_allowed_drift_score=1)

    # Act
    result = check.run(train, test, model)
    condition_result, *_ = check.conditions_decision(result)

    # Assert
    assert_that(condition_result, equal_condition_result(
        is_pass=True,
        details='Found model prediction PSI drift score of 0.79',
        name='Prediction drift score < 1'
    ))


def test_multiclass_proba(iris_split_dataset_and_model_rf):
    # Arrange
    train, test, model = iris_split_dataset_and_model_rf
    train = remove_label(train)
    test = remove_label(test)
    check = PredictionDrift(categorical_drift_method='PSI', numerical_drift_method='EMD',
                            max_num_categories=10, min_category_size_ratio=0,
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
    train = remove_label(train)
    test = remove_label(test)
    check = PredictionDrift(categorical_drift_method='PSI', numerical_drift_method='EMD', drift_mode='proba'
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
        name='Prediction drift score < 0.15',
        details='Found model prediction Earth Mover\'s Distance drift score of 0.23'
    ))


def test_multiclass_proba_reduce_aggregations(iris_split_dataset_and_model_rf):
    # Arrange
    train, test, model = iris_split_dataset_and_model_rf
    train = remove_label(train)
    test = remove_label(test)
    check = PredictionDrift(categorical_drift_method='PSI', numerical_drift_method='EMD',
                                     max_num_categories=10, min_category_size_ratio=0,
                                     drift_mode='proba', aggregation_method='weighted'
                                     ).add_condition_drift_score_less_than(max_allowed_drift_score=0.05)

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
        name='Prediction drift score < 0.05',
        details='Found 2 classes with model predicted probability Earth Mover\'s '
                'Distance drift score above threshold: 0.05.'
    ))
