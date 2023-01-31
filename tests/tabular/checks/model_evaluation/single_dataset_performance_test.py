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
"""Contains unit tests for the single dataset performance report check."""
from typing import List

from hamcrest import (assert_that, calling, close_to, has_entries, has_items, has_length,
                      instance_of, raises, equal_to, none)
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split

from tests.common import is_nan
from deepchecks.core import ConditionResult
from deepchecks.core.errors import DeepchecksNotSupportedError, DeepchecksValueError, ModelValidationError
from deepchecks.tabular.checks import SingleDatasetPerformance
from deepchecks.tabular.dataset import Dataset
from deepchecks.tabular.metric_utils.scorers import DEFAULT_MULTICLASS_SCORERS, DEFAULT_REGRESSION_SCORERS
from tests.base.utils import equal_condition_result


def test_dataset_wrong_input():
    bad_dataset = 'wrong_input'
    # Act & Assert
    assert_that(
        calling(SingleDatasetPerformance().run).with_args(bad_dataset, None, None),
        raises(DeepchecksValueError, 'non-empty instance of Dataset or DataFrame was expected, instead got str')
    )


def test_model_wrong_input(iris_labeled_dataset):
    bad_model = 'wrong_input'
    # Act & Assert
    assert_that(
        calling(SingleDatasetPerformance().run).with_args(iris_labeled_dataset, bad_model),
        raises(
            ModelValidationError,
            r'Model supplied does not meets the minimal interface requirements. Read more about .*')
    )


def test_dataset_no_label(iris_dataset_no_label, iris_adaboost):
    # Assert
    assert_that(
        calling(SingleDatasetPerformance().run).with_args(iris_dataset_no_label, iris_adaboost),
        raises(DeepchecksNotSupportedError,
               'Dataset does not contain a label column')
    )


def assert_multiclass_classification_result(result):
    for metric in DEFAULT_MULTICLASS_SCORERS.keys():
        metric_row = result.loc[result['Metric'] == metric]
        assert_that(metric_row['Value'].iloc[0], close_to(1, 0.3))


def test_missing_y_true_binary(missing_test_classes_binary_dataset_and_model):
    # Arrange
    _, test, model = missing_test_classes_binary_dataset_and_model
    check = SingleDatasetPerformance(scorers=['roc_auc'])
    # Act
    result = check.run(test, model)
    # Assert
    df = result.value
    assert_that(df, has_length(1))
    assert_that(df.loc[df['Metric'] == 'roc_auc']['Value'][0], none())


def test_classification(iris_split_dataset_and_model):
    # Arrange
    _, test, model = iris_split_dataset_and_model
    check = SingleDatasetPerformance()
    # Act
    result = check.run(test, model)
    # Assert
    assert_multiclass_classification_result(result.value)


def test_classification_new_classes_at_test(iris_split_dataset_and_model):
    # Arrange
    _, test, model = iris_split_dataset_and_model
    check = SingleDatasetPerformance(scorers=['precision_per_class', 'roc_auc_per_class'])
    test_new = test.data.copy()
    test_new.loc[test.n_samples] = [0, 1, 2, 3, 5]
    test = test.copy(test_new)
    # Act
    result = check.run(test, model).value
    # Assert
    assert_that(result.loc[3],
                has_entries({'Class': 5, 'Metric': 'precision', 'Value': is_nan()}))
    assert_that(result.loc[7],
                has_entries({'Class': 5, 'Metric': 'roc_auc', 'Value': is_nan()}))


def test_binary_classification_adult(adult_split_dataset_and_model):
    # Arrange
    _, test, model = adult_split_dataset_and_model
    check_binary = SingleDatasetPerformance()
    check_multiclass = SingleDatasetPerformance(scorers=['precision_per_class'])
    # Act
    result_binary = check_binary.run(test, model).value
    result_per_class = check_multiclass.run(test, model).value
    # Assert
    binary_precision = result_binary[result_binary['Metric'] == 'Precision'].iloc[0, 2]
    assert_that(binary_precision, close_to(0.79, 0.01))
    assert_that(result_per_class.iloc[1, 2], close_to(binary_precision, 0.01))


def test_classification_reduce(iris_split_dataset_and_model):
    # Arrange
    _, test, model = iris_split_dataset_and_model
    check = SingleDatasetPerformance()
    # Act
    result = check.run(test, model).reduce_output()
    # Assert
    assert_that(result, has_entries({'Accuracy': 0.92, 'Precision - Macro Average': close_to(0.92, 0.01)}))


def test_classification_reduce_macro(iris_split_dataset_and_model):
    # Arrange
    _, test, model = iris_split_dataset_and_model
    check = SingleDatasetPerformance(scorers={"awesome_f1": "f1_per_class", "awesome_f1_macro": "f1_macro"})
    # Act
    result = check.run(test, model).reduce_output()
    # Assert
    assert_that(result[('awesome_f1', '0')], close_to(1, 0.01))
    assert_that(result['awesome_f1_macro'], close_to(0.9131, 0.001))


def test_classification_without_display(iris_split_dataset_and_model):
    # Arrange
    _, test, model = iris_split_dataset_and_model
    check = SingleDatasetPerformance()
    # Act
    result = check.run(test, model, with_display=False)
    # Assert
    assert_multiclass_classification_result(result.value)
    assert_that(result.display, has_length(0))


def test_classification_binary(iris_dataset_single_class_labeled):
    # Arrange
    train, test = train_test_split(iris_dataset_single_class_labeled.data, test_size=0.33, random_state=42)
    train_ds = iris_dataset_single_class_labeled.copy(train)
    test_ds = iris_dataset_single_class_labeled.copy(test)
    clf = RandomForestClassifier(random_state=0)
    clf.fit(train_ds.data[train_ds.features], train_ds.data[train_ds.label_name])
    check = SingleDatasetPerformance()
    # Act
    result = check.run(test_ds, clf).value
    # Assert
    assert_that(max(result['Value']), close_to(1, 0.01))
    assert_that(list(result['Metric']), has_items('Accuracy', 'Recall'))


def test_classification_string_labels(iris_labeled_dataset):
    # Arrange
    check = SingleDatasetPerformance()
    replace_dict = {iris_labeled_dataset.label_name: {0: 'b', 1: 'e', 2: 'a'}}
    iris_labeled_dataset = Dataset(iris_labeled_dataset.data.replace(replace_dict),
                                   label=iris_labeled_dataset.label_name)

    iris_adaboost = AdaBoostClassifier(random_state=0)
    iris_adaboost.fit(iris_labeled_dataset.data[iris_labeled_dataset.features],
                      iris_labeled_dataset.data[iris_labeled_dataset.label_name])
    # Act
    result = check.run(iris_labeled_dataset, iris_adaboost).value
    # Assert
    assert_multiclass_classification_result(result)


def test_classification_nan_labels(iris_labeled_dataset, iris_adaboost):
    # Arrange
    check = SingleDatasetPerformance()
    data_with_nan = iris_labeled_dataset.data.copy()
    data_with_nan[iris_labeled_dataset.label_name].iloc[0] = float('nan')
    iris_labeled_dataset = Dataset(data_with_nan,
                                   label=iris_labeled_dataset.label_name)
    # Act
    result = check.run(iris_labeled_dataset, iris_adaboost).value
    # Assert
    assert_multiclass_classification_result(result)


def test_regression(diabetes_split_dataset_and_model):
    # Arrange
    train, test, model = diabetes_split_dataset_and_model
    check = SingleDatasetPerformance()
    # Act
    result = check.run(test, model).value
    # Assert
    for metric in DEFAULT_REGRESSION_SCORERS.keys():
        metric_col = result.loc[result['Metric'] == metric]
        assert_that(metric_col['Value'].iloc[0], instance_of(float))


def test_regression_positive_scorers(diabetes_split_dataset_and_model):
    # Arrange
    train, test, model = diabetes_split_dataset_and_model
    check = SingleDatasetPerformance(scorers=['mse', 'rmse', 'mae'])
    # Act
    result = check.run(test, model).value
    # Assert
    assert_that(result['Value'].iloc[0], close_to(3296, 1))
    assert_that(result['Value'].iloc[0], close_to(result['Value'].iloc[1] ** 2, 0.001))
    assert_that(result['Value'].iloc[2], close_to(45, 1))

    assert_that(check.greater_is_better(), equal_to(False))


def test_regression_positive_negative_compare(diabetes_split_dataset_and_model):
    # Arrange
    train, test, model = diabetes_split_dataset_and_model
    check = SingleDatasetPerformance(scorers=['mae', 'rmse', 'neg_mae', 'neg_rmse'])
    # Act
    result = check.run(test, model).value
    # Assert
    assert_that(result['Value'].iloc[0], close_to(-result['Value'].iloc[2], 1))
    assert_that(result['Value'].iloc[1], close_to(-result['Value'].iloc[3], 1))


def test_regression_reduced(diabetes_split_dataset_and_model):
    # Arrange
    _, test, model = diabetes_split_dataset_and_model
    check = SingleDatasetPerformance()
    # Act
    result = check.run(test, model).reduce_output()
    # Assert
    assert_that(result['Neg RMSE'], close_to(-57.412, 0.001))
    assert_that(result['Neg MAE'], close_to(-45.5645, 0.001))
    assert_that(result['R2'], close_to(0.427, 0.001))


def test_condition_all_score_not_passed(iris_split_dataset_and_model):
    # Arrange
    _, test, model = iris_split_dataset_and_model
    check = SingleDatasetPerformance().add_condition_greater_than(0.99)
    # Act
    result: List[ConditionResult] = check.conditions_decision(check.run(test, model))
    # Assert
    assert_that(result, has_items(
        equal_condition_result(is_pass=False,
                               details='Failed for metrics: [\'Accuracy\', \'Precision - Macro Average\', \'Recall - Macro Average\']',
                               name='Selected metrics scores are greater than 0.99')
    ))


def test_condition_score_not_passed_class_mode(iris_split_dataset_and_model):
    # Arrange
    _, test, model = iris_split_dataset_and_model
    check = SingleDatasetPerformance(scorers=['Precision_per_class']).add_condition_greater_than(0.99, class_mode=1)
    # Act
    result: List[ConditionResult] = check.conditions_decision(check.run(test, model))
    # Assert
    assert_that(result, has_items(
        equal_condition_result(is_pass=False,
                               details='Failed for metrics: [\'Precision\']',
                               name='Selected metrics scores are greater than 0.99')
    ))


def test_condition_any_score_passed(iris_split_dataset_and_model):
    # Arrange
    _, test, model = iris_split_dataset_and_model
    check = SingleDatasetPerformance().add_condition_greater_than(0.8, class_mode='any')
    # Act
    result: List[ConditionResult] = check.conditions_decision(check.run(test, model))
    # Assert
    assert_that(result, has_items(
        equal_condition_result(is_pass=True,
                               details='Passed for all of the metrics.',
                               name='Selected metrics scores are greater than 0.8')
    ))


def test_regression_alt_scores_list(diabetes_split_dataset_and_model):
    # Arrange
    _, test, model = diabetes_split_dataset_and_model
    check = SingleDatasetPerformance(scorers=['max_error', 'r2', 'neg_mean_absolute_error'])
    # Act
    result = check.run(test, model).reduce_output()
    # Assert
    assert_that(result['max_error'], close_to(-171.719, 0.001))
    assert_that(result['r2'], close_to(0.427, 0.001))
    assert_that(result['neg_mean_absolute_error'], close_to(-45.564, 0.001))
