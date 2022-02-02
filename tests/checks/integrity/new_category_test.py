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
from hamcrest import assert_that, calling, raises, has_length, equal_to, has_items

from deepchecks.tabular import Dataset
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.tabular.checks.integrity import CategoryMismatchTrainTest

from tests.checks.utils import equal_condition_result


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
    assert_that(result, has_length(0))


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
    assert_that(result, has_length(1))
    assert_that(result['col1']['n_new'], equal_to(1))
    assert_that(result['col1']['n_total_samples'], equal_to(4))


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
    assert_that(result, has_length(0))


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
    assert_that(result, has_length(1))
    assert_that(result['col1']['n_new'], equal_to(1))
    assert_that(result['col1']['n_total_samples'], equal_to(4))


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
    assert_that(result, has_length(1))
    assert_that(result['col1']['n_new'], equal_to(1))
    assert_that(result['col1']['n_total_samples'], equal_to(4))


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
    assert_that(result, has_length(0))


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
    assert_that(result, has_length(1))
    assert_that(result['col1']['n_new'], equal_to(1))
    assert_that(result['col1']['n_total_samples'], equal_to(4))


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
    assert_that(result, has_length(1))
    assert_that(result['col2']['n_new'], equal_to(1))
    assert_that(result['col2']['n_total_samples'], equal_to(11))


def test_condition_categories_fail():
    train_data = {'col1': ['a', 'b', 'c', 'd'], 'col2': ['a', 'b', 'c', 'd']}
    test_data = {'col1': ['a', 'b', 'c', 'e'], 'col2': ['a', 'b', 'c', 'd']}
    train_dataset = Dataset(pd.DataFrame(data=train_data, columns=['col1', 'col2']), cat_features=['col1', 'col2'])
    test_dataset = Dataset(pd.DataFrame(data=test_data, columns=['col1', 'col2']),
                           cat_features=['col1', 'col2'])

    # Arrange
    check = CategoryMismatchTrainTest().add_condition_new_categories_not_greater_than(0)

    # Act
    result = check.conditions_decision(check.run(train_dataset, test_dataset))

    assert_that(result, has_items(
        equal_condition_result(is_pass=False,
                               details='Found columns with number of new categories above threshold: {\'col1\': 1}',
                               name='Number of new category values is not greater than 0')
    ))


def test_condition_categories_pass():
    train_data = {'col1': ['a', 'b', 'c', 'd'], 'col2': ['a', 'b', 'c', 'd']}
    test_data = {'col1': ['a', 'b', 'c', 'e'], 'col2': ['a', 'b', 'c', 'd']}
    train_dataset = Dataset(pd.DataFrame(data=train_data, columns=['col1', 'col2']), cat_features=['col1', 'col2'])
    test_dataset = Dataset(pd.DataFrame(data=test_data, columns=['col1', 'col2']),
                           cat_features=['col1', 'col2'])

    # Arrange
    check = CategoryMismatchTrainTest().add_condition_new_categories_not_greater_than(1)

    # Act
    result = check.conditions_decision(check.run(train_dataset, test_dataset))

    assert_that(result, has_items(
        equal_condition_result(is_pass=True,
                               name='Number of new category values is not greater than 1')
    ))


def test_condition_count_fail():
    train_data = {'col1': ['a', 'b', 'c', 'd'], 'col2': ['a', 'b', 'c', 'd']}
    test_data = {'col1': ['a', 'b', 'c', 'e'], 'col2': ['a', 'b', 'c', 'd']}
    train_dataset = Dataset(pd.DataFrame(data=train_data, columns=['col1', 'col2']), cat_features=['col1', 'col2'])
    test_dataset = Dataset(pd.DataFrame(data=test_data, columns=['col1', 'col2']),
                           cat_features=['col1', 'col2'])

    # Arrange
    check = CategoryMismatchTrainTest().add_condition_new_category_ratio_not_greater_than(0.1)

    # Act
    result = check.conditions_decision(check.run(train_dataset, test_dataset))

    assert_that(result, has_items(
        equal_condition_result(
            is_pass=False,
            details='Found columns with ratio of new category samples above threshold: {\'col1\': \'25%\'}',
            name='Ratio of samples with a new category is not greater than 10%')
    ))


def test_condition_count_pass():
    train_data = {'col1': ['a', 'b', 'c', 'd'], 'col2': ['a', 'b', 'c', 'd']}
    test_data = {'col1': ['a', 'b', 'c', 'e'], 'col2': ['a', 'b', 'c', 'd']}
    train_dataset = Dataset(pd.DataFrame(data=train_data, columns=['col1', 'col2']), cat_features=['col1', 'col2'])
    test_dataset = Dataset(pd.DataFrame(data=test_data, columns=['col1', 'col2']),
                           cat_features=['col1', 'col2'])

    # Arrange
    check = CategoryMismatchTrainTest().add_condition_new_category_ratio_not_greater_than(0.3)

    # Act
    result = check.conditions_decision(check.run(train_dataset, test_dataset))

    assert_that(result, has_items(
        equal_condition_result(is_pass=True,
                               name='Ratio of samples with a new category is not greater than 30%')
    ))
