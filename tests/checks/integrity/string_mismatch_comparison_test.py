"""Contains unit tests for the string_mismatch check."""
import pandas as pd

from mlchecks.base import Dataset
from mlchecks.checks import StringMismatchComparison

from hamcrest import assert_that, has_length


def test_single_col_mismatch():
    # Arrange
    data = {'col1': ['Deep', 'deep', 'deep!!!', 'earth', 'foo', 'bar', 'foo?']}
    compared_data = {'col1': ['Deep', 'deep', '$deeP$', 'earth', 'foo', 'bar', 'foo?', '?deep']}

    # Act
    result = StringMismatchComparison().run(pd.DataFrame(data=data), pd.DataFrame(data=compared_data)).value

    # Assert - 1 column, 1 baseform are mismatch
    assert_that(result, has_length(1))


def test_mismatch_multi_column():
    # Arrange
    data = {'col1': ['Deep', 'deep', 'deep!!!', 'earth', 'foo', 'bar', 'foo?'],
            'col2': ['aaa', 'bbb', 'ddd', '><', '123', '111', '444']}
    compared_data = {'col1': ['Deep', 'deep', '$deeP$', 'earth', 'foo', 'bar', 'foo?', '?deep'],
                     'col2': ['aaa!', 'bbb!', 'ddd', '><', '123???', '123!', '__123__', '111']}

    # Act
    result = StringMismatchComparison().run(pd.DataFrame(data=data), pd.DataFrame(data=compared_data)).value

    # Assert - 2 columns, 4 baseforms are mismatch
    assert_that(result, has_length(4))


def test_no_mismatch():
    # Arrange
    data = {'col1': ['foo', 'bar', 'cat']}
    compared_data = {'col1': ['foo', 'foo', 'bar', 'bar', 'bar', 'dog?!']}

    # Act
    result = StringMismatchComparison().run(pd.DataFrame(data=data), pd.DataFrame(data=compared_data)).value

    # Assert
    assert_that(result, has_length(0))


def test_no_mismatch_on_numeric_column():
    # Arrange
    data = {'col1': ['foo', 'bar', 'cat'], 'col2': [10, 2.3, 1]}
    compared_data = {'col1': ['foo', 'foo', 'foo'], 'col2': [1, 2.30, 1.0]}

    # Act
    result = StringMismatchComparison().run(pd.DataFrame(data=data), pd.DataFrame(data=compared_data)).value

    # Assert
    assert_that(result, has_length(0))


def test_no_mismatch_on_numeric_string_column():
    # Arrange
    data = {'num_str': ['10', '2.3', '1']}
    compared_data = {'num_str': ['1', '2.30', '1.0']}

    # Act
    result = StringMismatchComparison().run(pd.DataFrame(data=data), pd.DataFrame(data=compared_data)).value

    # Assert
    assert_that(result, has_length(0))


def test_no_mismatch_on_one_column_numeric():
    # Arrange
    data = {'num_str': ['10', '2.3', '1']}
    compared_data = {'num_str': [1, 2.30, 1.0]}

    # Act
    result = StringMismatchComparison().run(pd.DataFrame(data=data), pd.DataFrame(data=compared_data)).value

    # Assert
    assert_that(result, has_length(0))

def test_fi_n_top(diabetes_split_dataset_and_model):
    train, val, clf = diabetes_split_dataset_and_model
    train = Dataset(train.data.copy(), label='target', cat_features=['sex'])
    val = Dataset(val.data.copy(), label='target', cat_features=['sex'])
    train.data.loc[train.data.index % 2 == 0, 'age'] = 'aaa'
    val.data.loc[val.data.index % 2 == 1, 'age'] = 'aaa!!'
    train.data.loc[train.data.index % 2 == 0, 'bmi'] = 'aaa'
    val.data.loc[val.data.index % 2 == 1, 'bmi'] = 'aaa!!'
    train.data.loc[train.data.index % 2 == 0, 'bp'] = 'aaa'
    val.data.loc[val.data.index % 2 == 1, 'bp'] = 'aaa!!'
    train.data.loc[train.data.index % 2 == 0, 'sex'] = 'aaa'
    val.data.loc[val.data.index % 2 == 1, 'sex'] = 'aaa!!'
    # Arrange
    check = StringMismatchComparison(n_top_columns=3)
    # Act
    result_ds = check.run(train, val, clf).value
    # Assert
    assert_that(result_ds, has_length(3))
