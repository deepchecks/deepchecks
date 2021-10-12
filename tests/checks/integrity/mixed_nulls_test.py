import pandas as pd
from hamcrest import *

from mlchecks.checks.integrity.mixed_nulls import mixed_nulls


def test_single_column_no_nulls():
    # Arrange
    data = {'col1': ['foo', 'bar', 'cat']}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = mixed_nulls(dataframe)
    # Assert
    assert_that(result.value, has_length(0))


def test_single_column_one_null_type():
    # Arrange
    data = {'col1': ['foo', 'bar', 'null', 'null']}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = mixed_nulls(dataframe)
    # Assert - Single null type is allowed so return is 0
    assert_that(result.value, has_length(0))


def test_single_column_two_null_types():
    # Arrange
    data = {'col1': ['foo', 'bar', 'null', 'nan', 'nan']}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = mixed_nulls(dataframe)
    # Assert
    assert_that(result.value, has_length(2))


def test_single_column_different_case_is_count_separated():
    # Arrange
    data = {'col1': ['foo', 'bar', 'Nan', 'nan', 'NaN']}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = mixed_nulls(dataframe)
    # Assert
    assert_that(result.value, has_length(3))


def test_single_column_nulls_with_special_characters():
    # Arrange
    data = {'col1': ['', '#@$', 'Nan!', '#nan', '<NaN>']}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = mixed_nulls(dataframe)
    # Assert
    assert_that(result.value, has_length(5))


def test_single_column_in_dataset_no_nulls():
    # Arrange
    data = {'col1': ['foo', 'bar', 'cat'], 'col2': ['nan', 'null', ''], 'col3': ['1', '2', '3']}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = mixed_nulls(dataframe, column='col1')
    # Assert
    assert_that(result.value, has_length(0))


def test_dataset_no_nulls():
    # Arrange
    data = {'col1': ['foo', 'bar', 'cat'], 'col2': ['foo', 'bar', 1], 'col3': [1, 2, 3]}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = mixed_nulls(dataframe)
    # Assert
    assert_that(result.value, has_length(0))


def test_dataset_1_column_nulls():
    # Arrange
    data = {'col1': ['foo', 'bar', 'null'], 'col2': ['foo', 'bar', 1], 'col3': [1, 2, 3]}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = mixed_nulls(dataframe)
    # Assert - Single null is allowed so still empty return
    assert_that(result.value, has_length(0))


def test_dataset_2_columns_single_nulls():
    # Arrange
    data = {'col1': ['foo', 'bar', 'null'], 'col2': ['Nan', 'bar', 1], 'col3': [1, 2, 3]}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = mixed_nulls(dataframe)
    # Assert - Single null is allowed so still empty return
    assert_that(result.value, has_length(0))
