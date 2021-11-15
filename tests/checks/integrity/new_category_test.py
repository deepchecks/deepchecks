"""Contains unit tests for the new_category_train_validation check"""

import pandas as pd
from deepchecks.base import Dataset
from deepchecks.utils import DeepchecksValueError
from deepchecks.checks.integrity import CategoryMismatchTrainTest
from hamcrest import assert_that, calling, raises, has_length, close_to


def test_dataset_wrong_input():
    x = 'wrong_input'
    # Act & Assert
    check = CategoryMismatchTrainTest()
    assert_that(calling(check.run).with_args(x, x),
                raises(DeepchecksValueError,
                       'dataset must be of type DataFrame or Dataset. instead got: str'))


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
    assert_that(result['col1'], close_to(0.25, 0.01))


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
    assert_that(result['col1'], close_to(0.25, 0.01))


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
    assert_that(result['col1'], close_to(0.25, 0.01))


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
    assert_that(result['col1'], close_to(0.25, 0.01))


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
    assert_that(result['col2'], close_to(0.09, 0.01))
