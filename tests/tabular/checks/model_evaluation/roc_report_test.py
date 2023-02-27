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
"""Contains unit tests for the roc_report check."""
import numpy as np
import pandas as pd
from hamcrest import assert_that, calling, close_to, greater_than, has_entries, has_items, has_length, raises
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from deepchecks.core.errors import DeepchecksNotSupportedError, DeepchecksValueError, ModelValidationError
from deepchecks.tabular.checks.model_evaluation import RocReport
from deepchecks.tabular.dataset import Dataset
from tests.base.utils import equal_condition_result


def test_dataset_wrong_input():
    bad_dataset = 'wrong_input'
    # Act & Assert
    assert_that(
        calling(RocReport().run).with_args(bad_dataset, None),
        raises(DeepchecksValueError, 'non-empty instance of Dataset or DataFrame was expected, instead got str')
    )


def test_dataset_no_label(iris_dataset_no_label, iris_adaboost):
    # Assert
    assert_that(
        calling(RocReport().run).with_args(iris_dataset_no_label, iris_adaboost),
        raises(DeepchecksNotSupportedError, 'Dataset does not contain a label column')
    )


def test_regression_model(diabetes_split_dataset_and_model):
    # Assert
    train, _, clf = diabetes_split_dataset_and_model
    assert_that(
        calling(RocReport().run).with_args(train, clf),
        raises(ModelValidationError, 'Check is irrelevant for regression tasks'))


def test_binary_classification(iris_binary_string_split_dataset_and_model):
    # Arrange
    train, _, clf = iris_binary_string_split_dataset_and_model
    # Act
    result = RocReport(excluded_classes=[]).run(train, clf)
    # Assert
    assert_that(result.value, has_length(2))
    assert_that(result.value, has_entries(a=close_to(0.9, 0.1), b=close_to(0.9, 0.1)))
    assert_that(result.display, has_length(greater_than(0)))


def test_binary_classification_without_display(iris_binary_string_split_dataset_and_model):
    # Arrange
    train, _, clf = iris_binary_string_split_dataset_and_model
    # Act
    result = RocReport().run(train, clf, with_display=False)
    # Assert
    assert_that(result.value, has_length(1))
    assert_that(result.value, has_entries(b=close_to(0.9, 0.1)))
    assert_that(result.display, has_length(0))


def test_model_info_object(iris_labeled_dataset, iris_adaboost):
    # Arrange
    check = RocReport()
    # Act X
    result = check.run(iris_labeled_dataset, iris_adaboost).value
    # Assert
    assert len(result) == 3  # iris has 3 targets
    for value in result.values():
        assert isinstance(value, np.float64)


def test_condition_ratio_more_than_not_passed(iris_clean):
    # Arrange
    clf = LogisticRegression(max_iter=1)
    x = iris_clean.data
    y = iris_clean.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=55)
    clf.fit(x_train, y_train)
    ds = Dataset(pd.concat([x_test, y_test], axis=1),
                 features=iris_clean.feature_names,
                 label='target')

    check = RocReport().add_condition_auc_greater_than(min_auc=0.8)

    # Act
    result = check.conditions_decision(check.run(ds, clf))

    assert_that(result, has_items(
        equal_condition_result(is_pass=False,
                               name='AUC score for all the classes is greater than 0.8',
                               details='Found classes with AUC below threshold: {1: \'0.71\'}')
    ))


def test_condition_ratio_more_than_passed(iris_clean):
    clf = LogisticRegression(max_iter=1)
    x = iris_clean.data
    y = iris_clean.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=55)
    clf.fit(x_train, y_train)
    ds = Dataset(pd.concat([x_test, y_test], axis=1),
                 features=iris_clean.feature_names,
                 label='target')

    check = RocReport().add_condition_auc_greater_than()

    result = check.conditions_decision(check.run(ds, clf))

    assert_that(result, has_items(
        equal_condition_result(is_pass=True,
                               details='All classes passed, minimum AUC found is 0.71 for class 1',
                               name='AUC score for all the classes is greater than 0.7')
    ))

    check = RocReport(excluded_classes=[1]).add_condition_auc_greater_than(min_auc=0.8)

    result = check.conditions_decision(check.run(ds, clf))

    assert_that(result, has_items(
        equal_condition_result(is_pass=True,
                               details='All classes passed, minimum AUC found is 1 for class 2',
                               name='AUC score for all the classes except: [1] is greater than 0.8')
    ))
