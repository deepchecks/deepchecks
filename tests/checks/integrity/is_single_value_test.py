from mlchecks.checks.integrity.is_single_value import *
import pandas as pd
from hamcrest import *

from mlchecks.utils import MLChecksValueError


def helper_test_df_and_result(df, expected_result_value):
    # Act
    result = is_single_value(df)

    # Assert
    assert_that(result.value==expected_result_value)    


def test_single_column_dataset_more_than_single_value():
    # Arrange
    df = pd.DataFrame({'a': [3, 4]})

    # Act & Assert
    helper_test_df_and_result(df, False)


def test_single_column_dataset_single_value():
    # Arrange
    df = pd.DataFrame({'a': ['b', 'b']})

    # Act & Assert
    helper_test_df_and_result(df, True)


# def test_model_info_wrong_input():
#     # Act
#     assert_that(calling(model_info).with_args('some string'),
#                 raises(MLChecksValueError, 'Model must inherit from one of supported models:'))
