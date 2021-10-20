"""Tests for Mixed Nulls check"""
import numpy as np
import pandas as pd

from hamcrest import assert_that, has_length

from mlchecks.checks.integrity.data_duplicates import data_duplicates, DataDuplicates

def test_single_column_no_nulls():
    # Arrange
    data = {'col1': ['foo', 'bar', 'cat']}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = data_duplicates(dataframe)
    # Assert
    assert_that(result.value, has_length(0))

