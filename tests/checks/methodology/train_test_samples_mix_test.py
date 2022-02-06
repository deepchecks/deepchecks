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
"""
Contains unit tests for the data_sample_leakage_report check
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from hamcrest import assert_that, calling, raises, equal_to, has_items, has_entry

from deepchecks.tabular import Dataset
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.tabular.checks.methodology import TrainTestSamplesMix
from tests.checks.utils import equal_condition_result


def test_dataset_wrong_input():
    x = 'wrong_input'
    # Act & Assert
    assert_that(
        calling(TrainTestSamplesMix().run).with_args(x, x),
        raises(DeepchecksValueError,
               'non-empty instance of Dataset or DataFrame was expected, instead got str'))


def test_no_leakage(iris_clean):
    x = iris_clean.data
    y = iris_clean.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=55)
    train_dataset = Dataset(pd.concat([x_train, y_train], axis=1),
                            features=iris_clean.feature_names,
                            label='target')

    test_df = pd.concat([x_test, y_test], axis=1)

    test_dataset = Dataset(test_df,
                           features=iris_clean.feature_names,
                           label='target')
    # Arrange
    check = TrainTestSamplesMix()
    # Act X
    result = check.run(test_dataset=test_dataset, train_dataset=train_dataset).value
    # Assert
    assert_that(result, has_entry('ratio', 0))


def test_leakage(iris_clean):
    x = iris_clean.data
    y = iris_clean.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=55)
    train_dataset = Dataset(pd.concat([x_train, y_train], axis=1),
                            features=iris_clean.feature_names,
                            label='target')

    test_df = pd.concat([x_test, y_test], axis=1)
    bad_test = test_df.append(train_dataset.data.iloc[[0, 1, 2, 3, 4]], ignore_index=True)

    test_dataset = Dataset(bad_test,
                           features=iris_clean.feature_names,
                           label='target')
    # Arrange
    check = TrainTestSamplesMix()
    # Act X
    result = check.run(test_dataset=test_dataset, train_dataset=train_dataset).value
    # Assert
    assert_that(result, has_entry('ratio', 0.1))


def test_nan():
    train_dataset = Dataset(pd.DataFrame({'col1': [1, 2, 3, np.nan], 'col2': [1, 2, 1, 1]}),
                            label='col2')
    test_dataset = Dataset(pd.DataFrame({'col1': [2, np.nan, np.nan, np.nan], 'col2': [1, 1, 2, 1]}),
                           label='col2')
    # Arrange
    check = TrainTestSamplesMix()
    # Act X
    result = check.run(test_dataset=test_dataset, train_dataset=train_dataset).value
    # Assert
    assert_that(result, has_entry('ratio', 0.5))


def test_condition_ratio_not_greater_than_not_passed(iris_clean):
    # Arrange
    x = iris_clean.data
    y = iris_clean.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=55)
    train_dataset = Dataset(pd.concat([x_train, y_train], axis=1),
                            features=iris_clean.feature_names,
                            label='target')

    test_df = pd.concat([x_test, y_test], axis=1)
    bad_test = test_df.append(train_dataset.data.iloc[[0, 1, 2, 3, 4]], ignore_index=True)

    test_dataset = Dataset(bad_test,
                           features=iris_clean.feature_names,
                           label='target')

    check = TrainTestSamplesMix().add_condition_duplicates_ratio_not_greater_than(max_ratio=0.09)

    # Act
    result = check.conditions_decision(check.run(train_dataset, test_dataset))

    assert_that(result, has_items(
        equal_condition_result(is_pass=False,
                               name='Percentage of test data samples that appear in train data '
                                    'not greater than 9%',
                               details='Percent of test data samples that appear in train data: '
                                       '10%')
    ))


def test_condition_ratio_not_greater_than_passed(diabetes_split_dataset_and_model):
    # Arrange
    train_ds, val_ds, clf = diabetes_split_dataset_and_model
    check = TrainTestSamplesMix().add_condition_duplicates_ratio_not_greater_than()

    # Act
    result = check.conditions_decision(check.run(train_ds, val_ds, clf))

    assert_that(result, has_items(
        equal_condition_result(is_pass=True,
                               name='Percentage of test data samples that appear in train data '
                                    'not greater than 10%')
    ))