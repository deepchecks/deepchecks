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
"""Contains unit tests for the performance report check."""
import re
from typing import List

from hamcrest import assert_that, calling, raises, close_to, has_items, instance_of
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split

from deepchecks.core import ConditionResult
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks.performance import PerformanceReport
from deepchecks.utils.metrics import MULTICLASS_SCORERS_NON_AVERAGE, DEFAULT_REGRESSION_SCORERS
from deepchecks.core.errors import DeepchecksValueError, DatasetValidationError, ModelValidationError, \
    DeepchecksNotSupportedError

from tests.checks.utils import equal_condition_result



def test_dataset_wrong_input():
    bad_dataset = 'wrong_input'
    # Act & Assert
    assert_that(
        calling(PerformanceReport().run).with_args(bad_dataset, None, None),
        raises(DeepchecksValueError, 'non-empty instance of Dataset or DataFrame was expected, instead got str')
    )


def test_model_wrong_input(iris_labeled_dataset):
    bad_model = 'wrong_input'
    # Act & Assert
    assert_that(
        calling(PerformanceReport().run).with_args(iris_labeled_dataset, iris_labeled_dataset,bad_model),
        raises(
            ModelValidationError,
            r'Model supplied does not meets the minimal interface requirements. Read more about .*')
    )


def test_dataset_no_label(iris_dataset_no_label, iris_adaboost):
    # Assert
    assert_that(
        calling(PerformanceReport().run).with_args(iris_dataset_no_label, iris_dataset_no_label, iris_adaboost),
        raises(DeepchecksNotSupportedError,
               'There is no label defined to use. Did you pass a DataFrame instead of a Dataset?')
    )


def test_dataset_no_shared_label(iris_labeled_dataset):
    # Assert
    iris_dataset_2 = Dataset(iris_labeled_dataset.data, label='sepal length (cm)')
    assert_that(
        calling(PerformanceReport().run).with_args(iris_labeled_dataset, iris_dataset_2, None),
        raises(DatasetValidationError, 'train and test requires to have and to share the same label')
    )


def assert_classification_result(result, dataset: Dataset):
    for dataset_name in ['Test', 'Train']:
        dataset_df = result.loc[result['Dataset'] == dataset_name]
        for class_name in dataset.classes:
            class_df = dataset_df.loc[dataset_df['Class'] == class_name]
            for metric in MULTICLASS_SCORERS_NON_AVERAGE.keys():
                metric_row = class_df.loc[class_df['Metric'] == metric]
                assert_that(metric_row['Value'].iloc[0], close_to(1, 0.3))


def test_classification(iris_split_dataset_and_model):
    # Arrange
    train, test, model = iris_split_dataset_and_model
    check = PerformanceReport()
    # Act X
    result = check.run(train, test, model).value
    # Assert
    assert_classification_result(result, test)


def test_classification_binary(iris_dataset_single_class_labeled):
    # Arrange
    train, test = train_test_split(iris_dataset_single_class_labeled.data, test_size=0.33, random_state=42)
    train_ds = iris_dataset_single_class_labeled.copy(train)
    test_ds = iris_dataset_single_class_labeled.copy(test)
    clf = RandomForestClassifier(random_state=0)
    clf.fit(train_ds.data[train_ds.features], train_ds.data[train_ds.label_name])
    check = PerformanceReport()

    # Act X
    result = check.run(train_ds, test_ds, clf).value
    # Assert
    assert_classification_result(result, test_ds)


def test_classification_string_labels(iris_labeled_dataset):
    # Arrange
    check = PerformanceReport()
    replace_dict = {iris_labeled_dataset.label_name: {0: 'b', 1: 'e', 2: 'a'}}
    iris_labeled_dataset = Dataset(iris_labeled_dataset.data.replace(replace_dict),
                                   label=iris_labeled_dataset.label_name)

    iris_adaboost = AdaBoostClassifier(random_state=0)
    iris_adaboost.fit(iris_labeled_dataset.data[iris_labeled_dataset.features],
                      iris_labeled_dataset.data[iris_labeled_dataset.label_name])
    # Act X
    result = check.run(iris_labeled_dataset, iris_labeled_dataset, iris_adaboost).value
    # Assert
    assert_classification_result(result, iris_labeled_dataset)


def test_classification_nan_labels(iris_labeled_dataset, iris_adaboost):
    # Arrange
    check = PerformanceReport()
    data_with_nan = iris_labeled_dataset.data.copy()
    data_with_nan[iris_labeled_dataset.label_name].iloc[0] = float('nan')
    iris_labeled_dataset = Dataset(data_with_nan,
                                   label=iris_labeled_dataset.label_name)
    # Act X
    result = check.run(iris_labeled_dataset, iris_labeled_dataset, iris_adaboost).value
    # Assert
    assert_classification_result(result, iris_labeled_dataset)


def test_regression(diabetes_split_dataset_and_model):
    # Arrange
    train, test, model = diabetes_split_dataset_and_model
    check = PerformanceReport()
    # Act X
    result = check.run(train, test, model).value
    # Assert
    for dataset in ['Test', 'Train']:
        dataset_col = result.loc[result['Dataset'] == dataset]
        for metric in DEFAULT_REGRESSION_SCORERS.keys():
            metric_col = dataset_col.loc[dataset_col['Metric'] == metric]
            assert_that(metric_col['Value'].iloc[0], instance_of(float))


def test_condition_min_score_not_passed(iris_split_dataset_and_model):
    # Arrange
    train, test, model = iris_split_dataset_and_model
    check = PerformanceReport().add_condition_test_performance_not_less_than(1)
    # Act X
    result: List[ConditionResult] = check.conditions_decision(check.run(train, test, model))
    # Assert
    assert_that(result, has_items(
        equal_condition_result(is_pass=False,
                               details=re.compile(r'Found metrics with scores below threshold:\n'),
                               name='Scores are not less than 1')
    ))


def test_condition_min_score_passed(iris_split_dataset_and_model):
    # Arrange
    train, test, model = iris_split_dataset_and_model
    check = PerformanceReport().add_condition_test_performance_not_less_than(0.5)
    # Act X
    result: List[ConditionResult] = check.conditions_decision(check.run(train, test, model))
    # Assert
    assert_that(result, has_items(
        equal_condition_result(is_pass=True,
                               name='Scores are not less than 0.5')
    ))


def test_condition_degradation_ratio_not_greater_than_not_passed(iris_split_dataset_and_model):
    # Arrange
    train, test, model = iris_split_dataset_and_model
    check = PerformanceReport().add_condition_train_test_relative_degradation_not_greater_than(0)
    # Act X
    result: List[ConditionResult] = check.conditions_decision(check.run(train, test, model))
    # Assert
    assert_that(result, has_items(
        equal_condition_result(is_pass=False,
                               details=re.compile(r'F1 for class 1 \(train=0.94 test=0.88\)'),
                               name='Train-Test scores relative degradation is not greater than 0')
    ))

def test_condition_degradation_ratio_not_greater_than_passed(iris_split_dataset_and_model):
    # Arrange
    train, test, model = iris_split_dataset_and_model
    check = PerformanceReport().add_condition_train_test_relative_degradation_not_greater_than(1)
    # Act X
    result: List[ConditionResult] = check.conditions_decision(check.run(train, test, model))
    # Assert
    assert_that(result, has_items(
        equal_condition_result(is_pass=True,
                               name='Train-Test scores relative degradation is not greater than 1')
    ))


def test_condition_class_performance_imbalance_ratio_not_greater_than_not_passed(iris_split_dataset_and_model):
    # Arrange
    train, test, model = iris_split_dataset_and_model
    check = PerformanceReport().add_condition_class_performance_imbalance_ratio_not_greater_than(0)
    # Act X
    result: List[ConditionResult] = check.conditions_decision(check.run(train, test, model))
    # Assert
    assert_that(result, has_items(
        equal_condition_result(is_pass=False,
                               details=re.compile('Relative ratio difference between highest and '
                                                  'lowest in Test dataset classes is 14.29%'),
                               name='Relative ratio difference between labels \'F1\' '
                                    'score is not greater than 0%')
    ))


def test_condition_class_performance_imbalance_ratio_not_greater_than_passed(iris_split_dataset_and_model):
    # Arrange
    train, test, model = iris_split_dataset_and_model
    check = PerformanceReport().add_condition_class_performance_imbalance_ratio_not_greater_than(1)
    # Act X
    result: List[ConditionResult] = check.conditions_decision(check.run(train, test, model))
    # Assert
    assert_that(result, has_items(
        equal_condition_result(is_pass=True,
                               name='Relative ratio difference between labels \'F1\' '
                               'score is not greater than 100%')
    ))
