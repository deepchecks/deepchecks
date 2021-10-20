"""The dataset_info check module."""
from typing import Union
import pandas as pd
from pandas_profiling import ProfileReport

from mlchecks import CheckResult, Dataset, SingleDatasetBaseCheck, ensure_dataframe_type

__all__ = ['dataset_info', 'DatasetInfo']


def dataset_info(dataset: Union[Dataset, pd.DataFrame]):
    """
    Summarize given dataset information based on pandas_profiling package.

    Args:
       dataset (Dataset): A dataset object
    Returns:
       CheckResult: value is tuple that represents the shape of the dataset

    Raises:
        MLChecksValueError: If the object is not of a supported type
    """
    dataset: pd.DataFrame = ensure_dataframe_type(dataset)

    def display():
        profile = ProfileReport(dataset, title='Dataset Report', explorative=True, minimal=True)
        profile.to_notebook_iframe()

    return CheckResult(dataset.shape, check=dataset_info, display=display)


class DatasetInfo(SingleDatasetBaseCheck):
    """Summarize given dataset information based on pandas_profiling package. Can be used inside `Suite`."""

    def run(self, dataset: Dataset, model=None) -> CheckResult:
        """
        Run the dataset_info check.

        Arguments:
            dataset: Dataset - The dataset object
            model: any = None - not used in the check

        Returns:
            the output of the dataset_info check
        """
        return dataset_info(dataset)

