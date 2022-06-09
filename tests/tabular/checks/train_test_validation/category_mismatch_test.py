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
"""Contains unit tests for the new_category_train_validation check"""

import pandas as pd
from hamcrest import assert_that, calling, equal_to, has_items, has_length, raises

from deepchecks.core.errors import DeepchecksValueError
from deepchecks.tabular.checks.train_test_validation import CategoryMismatchTrainTest
from deepchecks.tabular.dataset import Dataset
from tests.base.utils import equal_condition_result


def test_dataset_wrong_input():
    x = 'wrong_input'
    # Act & Assert
    check = CategoryMismatchTrainTest()
    assert_that(
        calling(check.run).with_args(x, x),
        raises(
            DeepchecksValueError,
            'non-empty instance of Dataset or DataFrame was expected, instead got str'
        )
    )


def test_no_new_category():
    train_data = {'col1': ['a', 'b', 'c']}
    test_data = {'col1': ['a', 'a', 'b', 'c']}
    train_dataset = Dataset(pd.DataFrame(data=train_data, columns=['col1']), cat_features=['col1'])
    test_dataset = Dataset(pd.DataFrame(data=test_data, columns=['col1']), cat_features=['col1'])

    # Arrange
    check = CategoryMismatchTrainTest()
    # Act X
    result = check.run(train_dataset=train_dataset, test_dataset=test_dataset).value
    # Assert
    assert_that(result, equal_to({'new_categories': {'col1': {}}, 'test_count': 4}))


def test_new_category():
    train_data = {'col1': ['a', 'b', 'c']}
    test_data = {'col1': ['a', 'b', 'c', 'd']}
    train_dataset = Dataset(pd.DataFrame(data=train_data, columns=['col1']), cat_features=['col1'])
    test_dataset = Dataset(pd.DataFrame(data=test_data, columns=['col1']), cat_features=['col1'])

    # Arrange
    check = CategoryMismatchTrainTest()
    # Act X
    result = check.run(train_dataset=train_dataset, test_dataset=test_dataset).value
    # Assert
    assert_that(result, equal_to({'new_categories': {'col1': {'d': 1}}, 'test_count': 4}))


def test_missing_category():
    train_data = {'col1': ['a', 'b', 'c', 'd']}
    test_data = {'col1': ['a', 'b', 'c']}
    train_dataset = Dataset(pd.DataFrame(data=train_data, columns=['col1']), cat_features=['col1'])
    test_dataset = Dataset(pd.DataFrame(data=test_data, columns=['col1']), cat_features=['col1'])

    # Arrange
    check = CategoryMismatchTrainTest()
    # Act X
    result = check.run(train_dataset=train_dataset, test_dataset=test_dataset).value
    # Assert
    assert_that(result, equal_to({'new_categories': {'col1': {}}, 'test_count': 3}))


def test_missing_new_category():
    train_data = {'col1': ['a', 'b', 'c', 'd']}
    test_data = {'col1': ['a', 'b', 'c', 'e']}
    train_dataset = Dataset(pd.DataFrame(data=train_data, columns=['col1']), cat_features=['col1'])
    test_dataset = Dataset(pd.DataFrame(data=test_data, columns=['col1']), cat_features=['col1'])

    # Arrange
    check = CategoryMismatchTrainTest()
    # Act X
    result = check.run(train_dataset=train_dataset, test_dataset=test_dataset).value
    # Assert
    assert_that(result, equal_to({'new_categories': {'col1': {'e': 1}}, 'test_count': 4}))


def test_multiple_categories():
    train_data = {'col1': ['a', 'b', 'c', 'd'], 'col2': ['a', 'b', 'c', 'd']}
    test_data = {'col1': ['a', 'b', 'c', 'e'], 'col2': ['a', 'b', 'c', 'd']}
    train_dataset = Dataset(pd.DataFrame(data=train_data, columns=['col1', 'col2']), cat_features=['col1', 'col2'])
    test_dataset = Dataset(pd.DataFrame(data=test_data, columns=['col1', 'col2']),
                           cat_features=['col1', 'col2'])

    # Arrange
    check = CategoryMismatchTrainTest()
    # Act X
    result = check.run(train_dataset=train_dataset, test_dataset=test_dataset).value
    # Assert
    assert_that(result, equal_to({'new_categories': {'col1': {'e': 1}, 'col2': {}}, 'test_count': 4}))


def test_ignore_column():
    train_data = {'col1': ['a', 'b', 'c', 'd'], 'col2': ['a', 'b', 'c', 'd']}
    test_data = {'col1': ['a', 'b', 'c', 'e'], 'col2': ['a', 'b', 'c', 'd']}
    train_dataset = Dataset(pd.DataFrame(data=train_data, columns=['col1', 'col2']), cat_features=['col1', 'col2'])
    test_dataset = Dataset(pd.DataFrame(data=test_data, columns=['col1', 'col2']),
                           cat_features=['col1', 'col2'])

    # Arrange
    check = CategoryMismatchTrainTest(ignore_columns='col1')
    # Act X
    result = check.run(train_dataset=train_dataset, test_dataset=test_dataset).value
    # Assert
    assert_that(result, equal_to({'new_categories': {'col2': {}}, 'test_count': 4}))


def test_specific_column():
    train_data = {'col1': ['a', 'b', 'c', 'd'], 'col2': ['a', 'b', 'c', 'd']}
    test_data = {'col1': ['a', 'b', 'c', 'e'], 'col2': ['a', 'b', 'c', 'd']}
    train_dataset = Dataset(pd.DataFrame(data=train_data, columns=['col1', 'col2']), cat_features=['col1', 'col2'])
    test_dataset = Dataset(pd.DataFrame(data=test_data, columns=['col1', 'col2']),
                           cat_features=['col1', 'col2'])

    # Arrange
    check = CategoryMismatchTrainTest(columns=['col1'])
    # Act X
    result = check.run(train_dataset=train_dataset, test_dataset=test_dataset).value
    # Assert
    assert_that(result, equal_to({'new_categories': {'col1': {'e': 1}}, 'test_count': 4}))


def test_nan(df_with_single_nans_in_different_rows, df_with_single_nan_in_col):
    train_dataset = Dataset(pd.DataFrame(data=df_with_single_nans_in_different_rows,
                                         columns=['col1', 'col2']),
                            cat_features=['col1', 'col2'])
    test_dataset = Dataset(pd.DataFrame(data=df_with_single_nan_in_col,
                                        columns=['col1', 'col2']),
                           cat_features=['col1', 'col2'])

    # Arrange
    check = CategoryMismatchTrainTest()
    # Act X
    result = check.run(train_dataset=train_dataset, test_dataset=test_dataset).value
    # Assert
    assert_that(result, equal_to({'new_categories': {'col1': {}, 'col2': {5: 1}}, 'test_count': 11}))


def test_condition_categories_fail():
    train_data = {'col1': ['a', 'b', 'c', 'd'], 'col2': ['a', 'b', 'c', 'd']}
    test_data = {'col1': ['a', 'b', 'c', 'e'], 'col2': ['a', 'b', 'c', 'd']}
    train_dataset = Dataset(pd.DataFrame(data=train_data, columns=['col1', 'col2']), cat_features=['col1', 'col2'])
    test_dataset = Dataset(pd.DataFrame(data=test_data, columns=['col1', 'col2']),
                           cat_features=['col1', 'col2'])

    # Arrange
    check = CategoryMismatchTrainTest().add_condition_new_categories_less_or_equal(0)

    # Act
    result = check.conditions_decision(check.run(train_dataset, test_dataset))

    assert_that(result, has_items(
        equal_condition_result(is_pass=False,
                               details='Found 1 out of 2 columns with number of new categories above threshold:'
                                       '\n{\'col1\': 1}',
                               name='Number of new category values is less or equal to 0')
    ))


def test_condition_categories_pass():
    train_data = {'col1': ['a', 'b', 'c', 'd'], 'col2': ['a', 'b', 'c', 'd']}
    test_data = {'col1': ['a', 'b', 'c', 'e'], 'col2': ['a', 'b', 'c', 'd']}
    train_dataset = Dataset(pd.DataFrame(data=train_data, columns=['col1', 'col2']), cat_features=['col1', 'col2'])
    test_dataset = Dataset(pd.DataFrame(data=test_data, columns=['col1', 'col2']),
                           cat_features=['col1', 'col2'])

    # Arrange
    check = CategoryMismatchTrainTest().add_condition_new_categories_less_or_equal(1)

    # Act
    result = check.conditions_decision(check.run(train_dataset, test_dataset))

    assert_that(result, has_items(
        equal_condition_result(is_pass=True,
                               details='Passed for 2 relevant columns. Top columns with new categories count:\n'
                                       '{\'col1\': 1}',
                               name='Number of new category values is less or equal to 1')
    ))


def test_condition_count_fail():
    train_data = {'col1': ['a', 'b', 'c', 'd'], 'col2': ['a', 'b', 'c', 'd']}
    test_data = {'col1': ['a', 'b', 'c', 'e'], 'col2': ['a', 'b', 'c', 'd']}
    train_dataset = Dataset(pd.DataFrame(data=train_data, columns=['col1', 'col2']), cat_features=['col1', 'col2'])
    test_dataset = Dataset(pd.DataFrame(data=test_data, columns=['col1', 'col2']),
                           cat_features=['col1', 'col2'])

    # Arrange
    check = CategoryMismatchTrainTest().add_condition_new_category_ratio_less_or_equal(0.1)

    # Act
    result = check.conditions_decision(check.run(train_dataset, test_dataset))

    assert_that(result, has_items(
        equal_condition_result(
            is_pass=False,
            details='Found 1 out of 2 columns with ratio of new category samples above threshold:'
                    '\n{\'col1\': \'25%\'}',
            name='Ratio of samples with a new category is less or equal to 10%')
    ))


def test_condition_count_pass():
    train_data = {'col1': ['a', 'b', 'c', 'd'], 'col2': ['a', 'b', 'c', 'd']}
    test_data = {'col1': ['a', 'b', 'c', 'e'], 'col2': ['a', 'b', 'c', 'd']}
    train_dataset = Dataset(pd.DataFrame(data=train_data, columns=['col1', 'col2']), cat_features=['col1', 'col2'])
    test_dataset = Dataset(pd.DataFrame(data=test_data, columns=['col1', 'col2']),
                           cat_features=['col1', 'col2'])

    # Arrange
    check = CategoryMismatchTrainTest().add_condition_new_category_ratio_less_or_equal(0.3)

    # Act
    result = check.conditions_decision(check.run(train_dataset, test_dataset))

    assert_that(result, has_items(
        equal_condition_result(is_pass=True,
                               details='Passed for 2 relevant columns. Top columns with new categories ratio:\n'
                                       '{\'col1\': \'25%\', \'col2\': \'0%\'}',
                               name='Ratio of samples with a new category is less or equal to 30%')
    ))
