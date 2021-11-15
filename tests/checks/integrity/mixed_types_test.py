"""Tests for Mixed Types check"""
import numpy as np
import pandas as pd

# Disable wildcard import check for hamcrest
#pylint: disable=unused-wildcard-import,wildcard-import
from hamcrest import assert_that, has_length, calling, raises
from mlchecks.base import Dataset

from mlchecks.checks.integrity.mixed_types import MixedTypes
from mlchecks.utils import MLChecksValueError


def test_single_column_no_mix():
    # Arrange
    data = {'col1': ['foo', 'bar', 'cat']}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = MixedTypes().run(dataframe)
    # Assert
    assert_that(result.value.columns, has_length(0))


def test_single_column_explicit_mix():
    # Arrange
    data = {'col1': [1, 'bar', 'cat']}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = MixedTypes().run(dataframe)
    # Assert
    assert_that(result.value.columns, has_length(1)) #2 types


def test_single_column_stringed_mix():
    # Arrange
    data = {'col1': ['1', 'bar', 'cat']}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = MixedTypes().run(dataframe)
    # Assert
    assert_that(result.value.columns, has_length(1))


def test_double_column_one_mix():
    # Arrange
    data = {'col1': ['1', 'bar', 'cat'], 'col2': [6, 66, 666.66]}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = MixedTypes().run(dataframe)
    # Assert
    assert_that(result.value.columns, has_length(1))


def test_double_column_ignored_mix():
    # Arrange
    data = {'col1': ['1', 'bar', 'cat'], 'col2': [6, 66, 666.66]}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = MixedTypes(ignore_columns=['col1']).run(dataframe)
    # Assert
    assert_that(result.value.columns, has_length(0))


def test_double_column_specific_mix():
    # Arrange
    data = {'col1': ['1', 'bar', 'cat'], 'col2': [6, 66, 666.66]}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = MixedTypes(columns=['col1']).run(dataframe)
    # Assert
    assert_that(result.value.columns, has_length(1))


def test_double_column_specific_and_ignored_mix():
    # Arrange
    data = {'col1': ['1', 'bar', 'cat'], 'col2': [6, 66, 666.66]}
    dataframe = pd.DataFrame(data=data)
    # Act & Assert
    check = MixedTypes(ignore_columns=['col1'], columns=['col1'])
    assert_that(calling(check.run).with_args(dataframe),
                raises(MLChecksValueError))


def test_double_column_double_mix():
    # Arrange
    data = {'col1': ['1', 'bar', 'cat'], 'col2': [6, '66.66.6', 666.66]}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = MixedTypes().run(dataframe)
    # Assert
    assert_that(result.value.columns, has_length(2))


def test_fi_n_top(diabetes_split_dataset_and_model):
    train, _, clf = diabetes_split_dataset_and_model
    train = Dataset(train.data.copy(), label='target', cat_features=['sex'])
    train.data.loc[train.data.index % 4 == 1, 'age'] = 'a'
    train.data.loc[train.data.index % 4 == 1, 'bmi'] = 'a'
    train.data.loc[train.data.index % 4 == 1, 'bp'] = 'a'
    train.data.loc[train.data.index % 4 == 1, 'sex'] = 'a'
    # Arrange
    check = MixedTypes(n_top_columns=3)
    # Act
    result_ds = check.run(train, clf).value
    # Assert
    assert_that(result_ds.columns, has_length(3))


def test_no_mix_nan():
    # Arrange
    data = {'col1': [np.nan, 'bar', 'cat'], 'col2': ['a', np.nan, np.nan]}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = MixedTypes().run(dataframe)
    # Assert
    assert_that(result.value.columns, has_length(0))


def test_mix_nan():
    # Arrange
    data = {'col1': [np.nan, '1', 'cat'], 'col2': ['7', np.nan, np.nan]}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = MixedTypes().run(dataframe)
    # Assert
    assert_that(result.value.columns, has_length(1))
