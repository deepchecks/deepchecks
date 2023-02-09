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
"""Contains unit tests for the new_category_train_validation check"""

import pandas as pd
from hamcrest import assert_that, calling, close_to, equal_to, greater_than, has_items, has_length, raises

from deepchecks.core.errors import DeepchecksValueError
from deepchecks.tabular.checks.train_test_validation import NewCategoryTrainTest
from deepchecks.tabular.dataset import Dataset
from tests.base.utils import equal_condition_result


def test_dataset_wrong_input():
    x = 'wrong_input'
    # Act & Assert
    check = NewCategoryTrainTest()
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
    check = NewCategoryTrainTest()
    # Act X
    result = check.run(train_dataset=train_dataset, test_dataset=test_dataset).value
    # Assert
    assert_that(max(result['# New Categories']), equal_to(0))


def test_new_category():
    train_data = {'col1': ['a', 'b', 'c']}
    test_data = {'col1': ['a', 'b', 'c', 'd']}
    train_dataset = Dataset(pd.DataFrame(data=train_data, columns=['col1']), cat_features=['col1'])
    test_dataset = Dataset(pd.DataFrame(data=test_data, columns=['col1']), cat_features=['col1'])

    # Arrange
    check = NewCategoryTrainTest(aggregation_method="none")
    # Act X
    result = check.run(train_dataset=train_dataset, test_dataset=test_dataset)
    # Assert
    assert_that(max(result.value['# New Categories']), equal_to(1))
    assert_that(max(result.value['Ratio of New Categories']), equal_to(0.25))
    assert_that(result.reduce_output()['col1'], equal_to(0.25))
    assert_that(result.display, has_length(greater_than(0)))


def test_new_category_without_display():
    train_data = {'col1': ['a', 'b', 'c']}
    test_data = {'col1': ['a', 'b', 'c', 'd']}
    train_dataset = Dataset(pd.DataFrame(data=train_data, columns=['col1']), cat_features=['col1'])
    test_dataset = Dataset(pd.DataFrame(data=test_data, columns=['col1']), cat_features=['col1'])

    # Arrange
    check = NewCategoryTrainTest()
    # Act X
    result = check.run(train_dataset=train_dataset, test_dataset=test_dataset, with_display=False)
    # Assert
    assert_that(max(result.value['# New Categories']), equal_to(1))
    assert_that(max(result.value['Ratio of New Categories']), equal_to(0.25))
    assert_that(result.reduce_output()['Max New Categories Ratio'], equal_to(0.25))
    assert_that(result.display, has_length(0))


def test_missing_category():
    train_data = {'col1': ['a', 'b', 'c', 'd']}
    test_data = {'col1': ['a', 'b', 'c']}
    train_dataset = Dataset(pd.DataFrame(data=train_data, columns=['col1']), cat_features=['col1'])
    test_dataset = Dataset(pd.DataFrame(data=test_data, columns=['col1']), cat_features=['col1'])

    # Arrange
    check = NewCategoryTrainTest()
    # Act X
    result = check.run(train_dataset=train_dataset, test_dataset=test_dataset).value
    # Assert
    assert_that(max(result['# New Categories']), equal_to(0))


def test_missing_new_category():
    train_data = {'col1': ['a', 'b', 'c', 'd']}
    test_data = {'col1': ['a', 'b', 'c', 'e']}
    train_dataset = Dataset(pd.DataFrame(data=train_data, columns=['col1']), cat_features=['col1'])
    test_dataset = Dataset(pd.DataFrame(data=test_data, columns=['col1']), cat_features=['col1'])

    # Arrange
    check = NewCategoryTrainTest()
    # Act X
    result = check.run(train_dataset=train_dataset, test_dataset=test_dataset).value
    # Assert
    assert_that(max(result['# New Categories']), equal_to(1))
    assert_that(max(result['Ratio of New Categories']), equal_to(0.25))


def test_multiple_categories():
    train_data = {'col1': ['a', 'b', 'c', 'd'], 'col2': ['a', 'b', 'c', 'd']}
    test_data = {'col1': ['a', 'b', 'c', 'e'], 'col2': ['a', 'b', 'c', 'd']}
    train_dataset = Dataset(pd.DataFrame(data=train_data, columns=['col1', 'col2']), cat_features=['col1', 'col2'])
    test_dataset = Dataset(pd.DataFrame(data=test_data, columns=['col1', 'col2']),
                           cat_features=['col1', 'col2'])

    # Arrange
    check = NewCategoryTrainTest()
    # Act X
    result = check.run(train_dataset=train_dataset, test_dataset=test_dataset).value
    # Assert
    assert_that(max(result['# New Categories']), equal_to(1))
    assert_that(max(result['Ratio of New Categories']), equal_to(0.25))
    assert_that(result['# New Categories']['col2'], equal_to(0))
    assert_that(result['Ratio of New Categories']['col2'], equal_to(0))


def test_ignore_column():
    train_data = {'col1': ['a', 'b', 'c', 'd'], 'col2': ['a', 'b', 'c', 'd']}
    test_data = {'col1': ['a', 'b', 'c', 'e'], 'col2': ['a', 'b', 'c', 'd']}
    train_dataset = Dataset(pd.DataFrame(data=train_data, columns=['col1', 'col2']), cat_features=['col1', 'col2'])
    test_dataset = Dataset(pd.DataFrame(data=test_data, columns=['col1', 'col2']),
                           cat_features=['col1', 'col2'])

    # Arrange
    check = NewCategoryTrainTest(ignore_columns='col1')
    # Act X
    result = check.run(train_dataset=train_dataset, test_dataset=test_dataset).value
    # Assert
    assert_that(max(result['# New Categories']), equal_to(0))
    assert_that(max(result['Ratio of New Categories']), equal_to(0))


def test_specific_column():
    train_data = {'col1': ['a', 'b', 'c', 'd'], 'col2': ['a', 'b', 'c', 'd']}
    test_data = {'col1': ['a', 'b', 'c', 'e'], 'col2': ['a', 'b', 'c', 'd']}
    train_dataset = Dataset(pd.DataFrame(data=train_data, columns=['col1', 'col2']), cat_features=['col1', 'col2'])
    test_dataset = Dataset(pd.DataFrame(data=test_data, columns=['col1', 'col2']),
                           cat_features=['col1', 'col2'])

    # Arrange
    check = NewCategoryTrainTest(columns=['col1'])
    # Act X
    result = check.run(train_dataset=train_dataset, test_dataset=test_dataset).value
    # Assert
    assert_that(max(result['# New Categories']), equal_to(1))
    assert_that(max(result['Ratio of New Categories']), equal_to(0.25))
    assert_that(result, has_length(1))


def test_nan(df_with_single_nans_in_different_rows, df_with_single_nan_in_col):
    train_dataset = Dataset(pd.DataFrame(data=df_with_single_nans_in_different_rows,
                                         columns=['col1', 'col2']),
                            cat_features=['col1', 'col2'])
    test_dataset = Dataset(pd.DataFrame(data=df_with_single_nan_in_col,
                                        columns=['col1', 'col2']),
                           cat_features=['col1', 'col2'])

    # Arrange
    check = NewCategoryTrainTest()
    # Act X
    result = check.run(train_dataset=train_dataset, test_dataset=test_dataset).value
    # Assert
    assert_that(max(result['# New Categories']), equal_to(1))
    assert_that(max(result['Ratio of New Categories']), close_to(0.09, 0.01))
    assert_that(result['New categories']['col2'], equal_to([5]))


def test_none():
    train = Dataset(pd.DataFrame(data={'cat': ['a', 'b', 'c']}), cat_features=['cat'])
    test = Dataset(pd.DataFrame(data={'cat': ['a', 'b', 'c', None]}), cat_features=['cat'])
    result = NewCategoryTrainTest().run(train, test).value
    assert_that(max(result['# New Categories']), equal_to(0))


def test_condition_categories_fail():
    train_data = {'col1': ['a', 'b', 'c', 'd'], 'col2': ['a', 'b', 'c', 'd']}
    test_data = {'col1': ['a', 'b', 'c', 'e'], 'col2': ['a', 'b', 'c', 'd']}
    train_dataset = Dataset(pd.DataFrame(data=train_data, columns=['col1', 'col2']), cat_features=['col1', 'col2'])
    test_dataset = Dataset(pd.DataFrame(data=test_data, columns=['col1', 'col2']), cat_features=['col1', 'col2'])

    # Arrange
    check = NewCategoryTrainTest().add_condition_new_categories_less_or_equal(0)

    # Act
    result = check.conditions_decision(check.run(train_dataset, test_dataset))

    assert_that(result, has_items(
        equal_condition_result(
            is_pass=False,
            details='Found 1 features with number of new categories above threshold: \n{\'col1\': 1}',
            name='Number of new category values is less or equal to 0'
        )
    ))


def test_condition_categories_pass():
    train_data = {'col1': ['a', 'b', 'c', 'd'], 'col2': ['a', 'b', 'c', 'd']}
    test_data = {'col1': ['a', 'b', 'c', 'e'], 'col2': ['a', 'b', 'c', 'd']}
    train_dataset = Dataset(pd.DataFrame(data=train_data, columns=['col1', 'col2']), cat_features=['col1', 'col2'])
    test_dataset = Dataset(pd.DataFrame(data=test_data, columns=['col1', 'col2']), cat_features=['col1', 'col2'])

    # Arrange
    check = NewCategoryTrainTest().add_condition_new_categories_less_or_equal(1)

    # Act
    result = check.conditions_decision(check.run(train_dataset, test_dataset))

    assert_that(result, has_items(
        equal_condition_result(is_pass=True,
                               details='Passed for 2 relevant features. Top features with new categories count: \n'
                                       '{\'col1\': 1}',
                               name='Number of new category values is less or equal to 1')
    ))


def test_condition_count_fail():
    train_data = {'col1': ['a', 'b', 'c', 'd'], 'col2': ['a', 'b', 'c', 'd']}
    test_data = {'col1': ['a', 'b', 'c', 'e'], 'col2': ['a', 'b', 'c', 'd']}
    train_dataset = Dataset(pd.DataFrame(data=train_data, columns=['col1', 'col2']), cat_features=['col1', 'col2'])
    test_dataset = Dataset(pd.DataFrame(data=test_data, columns=['col1', 'col2']), cat_features=['col1', 'col2'])

    # Arrange
    check = NewCategoryTrainTest().add_condition_new_category_ratio_less_or_equal(0.1)

    # Act
    result = check.conditions_decision(check.run(train_dataset, test_dataset))

    assert_that(result, has_items(
        equal_condition_result(
            is_pass=False,
            details='Found 1 features with ratio of new categories above threshold: '
                    '\n{\'col1\': \'25%\'}',
            name='Ratio of samples with a new category is less or equal to 10%')
    ))


def test_condition_count_pass():
    train_data = {'col1': ['a', 'b', 'c', 'd'], 'col2': ['a', 'b', 'c', 'd']}
    test_data = {'col1': ['a', 'b', 'c', 'e'], 'col2': ['a', 'b', 'c', 'd']}
    train_dataset = Dataset(pd.DataFrame(data=train_data, columns=['col1', 'col2']), cat_features=['col1', 'col2'])
    test_dataset = Dataset(pd.DataFrame(data=test_data, columns=['col1', 'col2']), cat_features=['col1', 'col2'])

    # Arrange
    check = NewCategoryTrainTest().add_condition_new_category_ratio_less_or_equal(0.3)

    # Act
    result = check.conditions_decision(check.run(train_dataset, test_dataset))

    assert_that(result, has_items(
        equal_condition_result(is_pass=True,
                               details='Passed for 2 relevant features. Top features with new categories ratio: '
                                       '\n{\'col1\': \'25%\'}',
                               name='Ratio of samples with a new category is less or equal to 30%')
    ))
