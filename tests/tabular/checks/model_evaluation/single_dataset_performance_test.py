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
"""Contains unit tests for the single dataset performance report check."""
from typing import List

from hamcrest import assert_that, calling, close_to, has_items, has_length, instance_of, raises, \
    has_entries
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split

from deepchecks.core import ConditionResult
from deepchecks.core.errors import (DeepchecksNotSupportedError, DeepchecksValueError,
                                    ModelValidationError)
from deepchecks.tabular.checks import SingleDatasetPerformance
from deepchecks.tabular.dataset import Dataset
from deepchecks.tabular.metric_utils.scorers import DEFAULT_REGRESSION_SCORERS, MULTICLASS_SCORERS_NON_AVERAGE
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


def assert_classification_result(result, dataset: Dataset):
    for class_name in dataset.classes:
        class_df = result.loc[result['Class'] == class_name]
        for metric in MULTICLASS_SCORERS_NON_AVERAGE.keys():
            metric_row = class_df.loc[class_df['Metric'] == metric]
            assert_that(metric_row['Value'].iloc[0], close_to(1, 0.3))


def test_classification(iris_split_dataset_and_model):
    # Arrange
    _, test, model = iris_split_dataset_and_model
    check = SingleDatasetPerformance()
    # Act
    result = check.run(test, model)
    # Assert
    assert_classification_result(result.value, test)


def test_classification_reduce(iris_split_dataset_and_model):
    # Arrange
    _, test, model = iris_split_dataset_and_model
    check = SingleDatasetPerformance()
    # Act
    result = check.run(test, model).reduce_output()
    # Assert
    assert_that(result, has_entries({'F1_0': 1, 'Precision_2': 1, 'Recall_1': 1}))


def test_classification_reduce_macro(iris_split_dataset_and_model):
    # Arrange
    _, test, model = iris_split_dataset_and_model
    check = SingleDatasetPerformance(scorers={"awesome_f1": "f1_per_class", "awesome_f1_macro": "f1_macro"})
    # Act
    result = check.run(test, model).reduce_output()
    # Assert
    assert_that(result, has_entries({'awesome_f1_0': 1, 'awesome_f1_macro': close_to(0.9131, 0.001)}))


def test_classification_without_display(iris_split_dataset_and_model):
    # Arrange
    _, test, model = iris_split_dataset_and_model
    check = SingleDatasetPerformance()
    # Act
    result = check.run(test, model, with_display=False)
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
    check = SingleDatasetPerformance()

    # Act
    result = check.run(test_ds, clf).value
    # Assert
    assert_classification_result(result, test_ds)


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
    assert_classification_result(result, iris_labeled_dataset)


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
    assert_classification_result(result, iris_labeled_dataset)


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
    check = SingleDatasetPerformance().add_condition_greater_than(0.8)
    # Act
    result: List[ConditionResult] = check.conditions_decision(check.run(test, model))
    # Assert
    assert_that(result, has_items(
        equal_condition_result(is_pass=False,
                               details='Failed for metrics: [\'Precision\', \'Recall\']',
                               name='Selected metrics scores are greater than 0.8')
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
