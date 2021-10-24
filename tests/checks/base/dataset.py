"""Contains unit tests for the Dataset class."""
from mlchecks import Dataset
from mlchecks.utils import MLChecksValueError
from hamcrest import assert_that, calling, raises


def test_validate_dataset_empty_df(empty_df):
    assert_that(calling(Dataset.validate_dataset).with_args(Dataset(empty_df), 'test_function'),
                raises(MLChecksValueError, 'function test_function required a non-empty dataset'))


def test_validate_dataset_or_dataframe_empty_df(empty_df):
    assert_that(calling(Dataset.validate_dataset_or_dataframe).with_args(empty_df),
                raises(MLChecksValueError, 'dataset cannot be empty'))
