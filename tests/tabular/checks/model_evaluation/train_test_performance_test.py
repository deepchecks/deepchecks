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
"""Contains unit tests for the performance report check."""
import re
from typing import List

import numpy as np
from hamcrest import assert_that, calling, close_to, equal_to, greater_than, has_items, has_length, instance_of, raises
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import jaccard_score, make_scorer
from sklearn.model_selection import train_test_split

from deepchecks.core import ConditionResult
from deepchecks.core.errors import (DatasetValidationError, DeepchecksNotSupportedError, DeepchecksValueError,
                                    ModelValidationError)
from deepchecks.tabular.checks.model_evaluation import TrainTestPerformance
from deepchecks.tabular.dataset import Dataset
from deepchecks.tabular.metric_utils.scorers import DEFAULT_REGRESSION_SCORERS, MULTICLASS_SCORERS_NON_AVERAGE
from tests.base.utils import equal_condition_result


def extract_metric(result, dataset, metric):
    dataset_col = result.loc[result['Dataset'] == dataset]
    metric_col = dataset_col.loc[dataset_col['Metric'] == metric]
    if len(metric_col) == 1:
        return metric_col['Value'].iloc[0]
    else:
        return metric_col['Value'].mean()


def test_dataset_wrong_input():
    bad_dataset = 'wrong_input'
    # Act & Assert
    assert_that(
        calling(TrainTestPerformance().run).with_args(bad_dataset, None, None),
        raises(DeepchecksValueError, 'non-empty instance of Dataset or DataFrame was expected, instead got str')
    )


def test_model_wrong_input(iris_labeled_dataset):
    bad_model = 'wrong_input'
    # Act & Assert
    assert_that(
        calling(TrainTestPerformance().run).with_args(iris_labeled_dataset, iris_labeled_dataset, bad_model),
        raises(
            ModelValidationError,
            r'Model supplied does not meets the minimal interface requirements. Read more about .*')
    )


def test_dataset_no_label(iris_dataset_no_label, iris_adaboost):
    # Assert
    assert_that(
        calling(TrainTestPerformance().run).with_args(iris_dataset_no_label, iris_dataset_no_label, iris_adaboost),
        raises(DeepchecksNotSupportedError,
               'Dataset does not contain a label column')
    )


def test_dataset_no_shared_label(iris_labeled_dataset):
    # Assert
    iris_dataset_2 = Dataset(iris_labeled_dataset.data, label='sepal length (cm)')
    assert_that(
        calling(TrainTestPerformance().run).with_args(iris_labeled_dataset, iris_dataset_2, None),
        raises(DatasetValidationError, 'train and test requires to have and to share the same label')
    )


def assert_classification_result(result, dataset: Dataset):
    for dataset_name in ['Test', 'Train']:
        dataset_df = result.loc[result['Dataset'] == dataset_name]
        for class_name in dataset.classes_in_label_col:
            class_df = dataset_df.loc[dataset_df['Class'] == class_name]
            for metric in MULTICLASS_SCORERS_NON_AVERAGE.keys():
                metric_row = class_df.loc[class_df['Metric'] == metric]
                assert_that(metric_row['Value'].iloc[0], close_to(1, 0.3))


def test_classification(iris_split_dataset_and_model):
    # Arrange
    train, test, model = iris_split_dataset_and_model
    check = TrainTestPerformance()
    # Act
    result = check.run(train, test, model)
    # Assert
    assert_classification_result(result.value, test)
    assert_that(result.display, has_length(greater_than(0)))


def test_classification_without_display(iris_split_dataset_and_model):
    # Arrange
    train, test, model = iris_split_dataset_and_model
    check = TrainTestPerformance()
    # Act
    result = check.run(train, test, model, with_display=False)
    # Assert
    assert_classification_result(result.value, test)
    assert_that(result.display, has_length(0))


def test_classification_binary(iris_dataset_single_class_labeled):
    # Arrange
    train, test = train_test_split(iris_dataset_single_class_labeled.data, test_size=0.33, random_state=42)
    train_ds = iris_dataset_single_class_labeled.copy(train)
    test_ds = iris_dataset_single_class_labeled.copy(test)
    clf = RandomForestClassifier(random_state=0)
    clf.fit(train_ds.data[train_ds.features], train_ds.data[train_ds.label_name])
    check = TrainTestPerformance()

    # Act
    result = check.run(train_ds, test_ds, clf).value
    # Assert
    assert_classification_result(result, test_ds)


def test_classification_string_labels(iris_labeled_dataset):
    # Arrange
    check = TrainTestPerformance()
    replace_dict = {iris_labeled_dataset.label_name: {0: 'b', 1: 'e', 2: 'a'}}
    iris_labeled_dataset = Dataset(iris_labeled_dataset.data.replace(replace_dict),
                                   label=iris_labeled_dataset.label_name)

    iris_adaboost = AdaBoostClassifier(random_state=0)
    iris_adaboost.fit(iris_labeled_dataset.data[iris_labeled_dataset.features],
                      iris_labeled_dataset.data[iris_labeled_dataset.label_name])
    # Act
    result = check.run(iris_labeled_dataset, iris_labeled_dataset, iris_adaboost).value
    # Assert
    assert_classification_result(result, iris_labeled_dataset)


def test_classification_nan_labels(iris_labeled_dataset, iris_adaboost):
    # Arrange
    check = TrainTestPerformance()
    data_with_nan = iris_labeled_dataset.data.copy()
    data_with_nan[iris_labeled_dataset.label_name].iloc[0] = float('nan')
    iris_labeled_dataset = Dataset(data_with_nan,
                                   label=iris_labeled_dataset.label_name)
    # Act
    result = check.run(iris_labeled_dataset, iris_labeled_dataset, iris_adaboost).value
    # Assert
    assert_classification_result(result, iris_labeled_dataset)


def test_regression(diabetes_split_dataset_and_model):
    # Arrange
    train, test, model = diabetes_split_dataset_and_model
    check = TrainTestPerformance()
    # Act
    result = check.run(train, test, model).value
    # Assert
    for dataset in ['Test', 'Train']:
        dataset_col = result.loc[result['Dataset'] == dataset]
        for metric in DEFAULT_REGRESSION_SCORERS.keys():
            metric_col = dataset_col.loc[dataset_col['Metric'] == metric]
            assert_that(metric_col['Value'].iloc[0], instance_of(float))

def test_regression_reduced(diabetes_split_dataset_and_model):
    # Arrange
    train, test, model = diabetes_split_dataset_and_model
    check = TrainTestPerformance()
    # Act
    result = check.run(train, test, model).value
    # Assert
    assert_that(extract_metric(result, 'Test', 'Neg RMSE'), close_to(-57.412, 0.001))
    assert_that(extract_metric(result, 'Test', 'Neg MAE'), close_to(-45.5645, 0.001))
    assert_that(extract_metric(result, 'Test', 'R2'), close_to(0.427, 0.001))


def test_classification_reduced(iris_split_dataset_and_model):
    # Arrange
    train, test, model = iris_split_dataset_and_model
    check = TrainTestPerformance()
    # Act
    result = check.run(train, test, model).value
    # Assert
    assert_that(extract_metric(result, 'Test', 'F1'), close_to(0.913, 0.001))
    assert_that(extract_metric(result, 'Test', 'Precision'), close_to(0.929, 0.001))
    assert_that(extract_metric(result, 'Test', 'Recall'), close_to(0.916, 0.001))


def test_condition_min_score_not_passed(iris_split_dataset_and_model):
    # Arrange
    train, test, model = iris_split_dataset_and_model
    check = TrainTestPerformance().add_condition_test_performance_greater_than(1)
    # Act
    result: List[ConditionResult] = check.conditions_decision(check.run(train, test, model))
    # Assert
    assert_that(result, has_items(
        equal_condition_result(is_pass=False,
                               details='Found 9 scores below threshold.\nFound minimum score '
                                       'for Recall metric of value 0.75 for class 2.',
                               name='Scores are greater than 1')
    ))


def test_condition_min_score_passed(iris_split_dataset_and_model):
    # Arrange
    train, test, model = iris_split_dataset_and_model
    check = TrainTestPerformance().add_condition_test_performance_greater_than(0.5)
    # Act
    result: List[ConditionResult] = check.conditions_decision(check.run(train, test, model))
    # Assert
    assert_that(result, has_items(
        equal_condition_result(is_pass=True,
                               details='Found minimum score for Recall metric of value 0.75 for class 2.',
                               name='Scores are greater than 0.5')
    ))


def test_condition_degradation_ratio_less_than_not_passed(iris_split_dataset_and_model):
    # Arrange
    train, test, model = iris_split_dataset_and_model
    check = TrainTestPerformance().add_condition_train_test_relative_degradation_less_than(0)
    # Act
    result: List[ConditionResult] = check.conditions_decision(check.run(train, test, model))
    # Assert
    assert_that(result, has_items(
        equal_condition_result(is_pass=False,
                               details=r'7 scores failed. Found max degradation of 17.74% for metric Recall and class 2.',
                               name='Train-Test scores relative degradation is less than 0')
    ))


def test_condition_degradation_ratio_less_than_passed(iris_split_dataset_and_model):
    # Arrange
    train, test, model = iris_split_dataset_and_model
    check = TrainTestPerformance().add_condition_train_test_relative_degradation_less_than(1)
    # Act
    result: List[ConditionResult] = check.conditions_decision(check.run(train, test, model))
    # Assert
    assert_that(result, has_items(
        equal_condition_result(is_pass=True,
                               details=r'Found max degradation of 17.74% for metric Recall and class 2.',
                               name='Train-Test scores relative degradation is less than 1')
    ))


def test_condition_degradation_ratio_less_than_passed_regression(diabetes_split_dataset_and_model):
    # Arrange
    train, test, model = diabetes_split_dataset_and_model
    check = TrainTestPerformance().add_condition_train_test_relative_degradation_less_than(1)
    # Act
    result: List[ConditionResult] = check.conditions_decision(check.run(train, test, model))
    # Assert
    assert_that(result, has_items(
        equal_condition_result(is_pass=True,
                               details=r'Found max degradation of 94.98% for metric Neg MAE',
                               name='Train-Test scores relative degradation is less than 1')
    ))


def test_condition_degradation_ratio_less_than_not_passed_regression(diabetes_split_dataset_and_model):
    # Arrange
    train, test, model = diabetes_split_dataset_and_model
    check = TrainTestPerformance().add_condition_train_test_relative_degradation_less_than(0)
    # Act
    result: List[ConditionResult] = check.conditions_decision(check.run(train, test, model))
    # Assert
    assert_that(result, has_items(
        equal_condition_result(is_pass=False,
                               details=r'3 scores failed. Found max degradation of 94.98% for metric Neg MAE',
                               name='Train-Test scores relative degradation is less than 0')
    ))


def test_condition_class_performance_imbalance_ratio_less_than_not_passed(iris_split_dataset_and_model):
    # ArrangeF
    train, test, model = iris_split_dataset_and_model
    check = TrainTestPerformance().add_condition_class_performance_imbalance_ratio_less_than(0)
    # Act
    result: List[ConditionResult] = check.conditions_decision(check.run(train, test, model))
    # Assert
    assert_that(result, has_items(
        equal_condition_result(is_pass=False,
                               details=re.compile(
                                   'Relative ratio difference between highest and lowest in Test dataset classes is 14.29%'),
                               name='Relative ratio difference between labels \'F1\' score is less than 0%')
    ))


def test_condition_class_performance_imbalance_ratio_less_than_passed(iris_split_dataset_and_model):
    # Arrange
    train, test, model = iris_split_dataset_and_model
    check = TrainTestPerformance().add_condition_class_performance_imbalance_ratio_less_than(1)
    # Act
    result: List[ConditionResult] = check.conditions_decision(check.run(train, test, model))
    # Assert
    assert_that(result, has_items(
        equal_condition_result(is_pass=True,
                               details=re.compile(
                                   'Relative ratio difference between highest and lowest in Test dataset classes is 14.29%'),
                               name='Relative ratio difference between labels \'F1\' score is less than 100%')
    ))


def test_classification_alt_scores_list(iris_split_dataset_and_model):
    # Arrange
    train, test, model = iris_split_dataset_and_model
    check = TrainTestPerformance(scorers=['recall_per_class',
                                 'f1_per_class', make_scorer(jaccard_score, average=None)])
    # Act
    result = check.run(train, test, model).value
    # Assert
    assert_that(extract_metric(result, 'Test', 'f1'), close_to(0.913, 0.001))
    assert_that(extract_metric(result, 'Test', 'recall'), close_to(0.916, 0.001))
    assert_that(extract_metric(result, 'Test', 'jaccard_score'), close_to(0.846, 0.001))


def test_classification_deepchecks_scorers(iris_split_dataset_and_model):
    # Arrange
    train, test, model = iris_split_dataset_and_model
    check = TrainTestPerformance(scorers=['fpr_per_class', 'fnr_per_class', 'specificity_per_class', 'fnr_macro',
                                          'roc_auc_per_class'])
    # Act
    check_result = check.run(train, test, model)
    check_value = check_result.value

    per_class_df = check_value[check_value['Metric'] == 'roc_auc']
    assert_that(per_class_df.loc[per_class_df.Dataset == 'Train', 'Class'].values[1], equal_to(1.))
    assert_that(per_class_df.loc[per_class_df.Dataset == 'Train', 'Value'].values[1], close_to(0.997, 0.001))


def test_regression_alt_scores_list(diabetes_split_dataset_and_model):
    # Arrange
    train, test, model = diabetes_split_dataset_and_model
    check = TrainTestPerformance(scorers=['max_error', 'r2', 'neg_mean_absolute_error'])
    # Act
    result = check.run(train, test, model).value
    # Assert
    assert_that(extract_metric(result, 'Test', 'max_error'), close_to(-171.719, 0.001))
    assert_that(extract_metric(result, 'Test', 'r2'), close_to(0.427, 0.001))
    assert_that(extract_metric(result, 'Test', 'neg_mean_absolute_error'), close_to(-45.564, 0.001))


def test_classification_alt_scores_per_class_and_macro(iris_split_dataset_and_model):
    # Arrange
    train, test, model = iris_split_dataset_and_model
    check = TrainTestPerformance(scorers=['recall_per_class', 'f1_per_class', 'f1_macro', 'recall_micro'])
    # Act
    result = check.run(train, test, model).value
    # Assert
    assert_that(extract_metric(result, 'Test', 'f1'), close_to(0.913, 0.001))
    assert_that(extract_metric(result, 'Test', 'f1_macro'), close_to(0.913, 0.001))
    assert_that(extract_metric(result, 'Test', 'recall'), close_to(0.916, 0.001))
    assert_that(extract_metric(result, 'Test', 'recall_micro'), close_to(0.92, 0.001))
