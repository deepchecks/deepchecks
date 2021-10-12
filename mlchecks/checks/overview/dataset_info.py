"""The dataset_info check module."""
from typing import Union
import pandas as pd
from mlchecks import CheckResult, Dataset, SingleDatasetBaseCheck
from mlchecks.base.dataset import validate_dataset
from mlchecks.utils import is_notebook

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
    dataset = validate_dataset(dataset)

    if is_notebook():
        html = dataset.get_profile().to_notebook_iframe()
    else:
        html = dataset.to_html()

    return CheckResult(dataset.shape, display={'text/html': html})


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

