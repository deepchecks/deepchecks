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
"""Contains unit tests for the roc_report check."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from hamcrest import assert_that, calling, raises, has_items, has_entries, has_length, close_to

from deepchecks.tabular import Dataset
from deepchecks.tabular.checks.performance import RocReport
from deepchecks.core.errors import DeepchecksValueError, ModelValidationError, DeepchecksNotSupportedError
from tests.checks.utils import equal_condition_result


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
        raises(DeepchecksNotSupportedError, 'There is no label defined to use. Did you pass a DataFrame instead '
                                            'of a Dataset?')
    )


def test_regression_model(diabetes_split_dataset_and_model):
    # Assert
    train, _, clf = diabetes_split_dataset_and_model
    assert_that(
        calling(RocReport().run).with_args(train, clf),
        raises(
            ModelValidationError,
            r'Check is relevant for models of type '
            r'\[\'multiclass\', \'binary\'\], but received model of type \'regression\'')
    )


def test_binary_classification(iris_binary_string_split_dataset_and_model):
    # Arrange
    train, _, clf = iris_binary_string_split_dataset_and_model
    # Act
    result = RocReport().run(train, clf).value
    # Assert
    assert_that(result, has_length(2))
    assert_that(result, has_entries(a=close_to(0.9, 0.1), b=close_to(0.9, 0.1)))


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

    check = RocReport().add_condition_auc_not_less_than(min_auc=0.8)

    # Act
    result = check.conditions_decision(check.run(ds, clf))

    assert_that(result, has_items(
        equal_condition_result(is_pass=False,
                               name='AUC score for all the classes is not less than 0.8',
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

    check = RocReport().add_condition_auc_not_less_than()

    result = check.conditions_decision(check.run(ds, clf))

    assert_that(result, has_items(
        equal_condition_result(is_pass=True,
                               name='AUC score for all the classes is not less than 0.7')
    ))

    check = RocReport(excluded_classes=[1]).add_condition_auc_not_less_than(min_auc=0.8)

    result = check.conditions_decision(check.run(ds, clf))

    assert_that(result, has_items(
        equal_condition_result(is_pass=True,
                               name='AUC score for all the classes except: [1] is not less than 0.8')
    ))
