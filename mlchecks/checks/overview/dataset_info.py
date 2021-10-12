"""The dataset_info check module."""
from typing import Union
import pandas as pd
from mlchecks import CheckResult, Dataset, SingleDatasetBaseCheck
from mlchecks.utils import is_notebook, MLChecksValueError

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

    if not isinstance(dataset, pd.DataFrame):
        raise MLChecksValueError("dataset_info check must receive a DataFrame or a Dataset object")

    # If we receive a DataFrame but not a Dataset, convert it
    if not isinstance(dataset, Dataset):
        d = Dataset(dataset)
    else:
        d = dataset

    if is_notebook():
        html = d.get_profile().to_notebook_iframe()
    else:
        html = d.to_html()

    return CheckResult(dataset.shape,
                       display={'text/html': html})


class DatasetInfo(SingleDatasetBaseCheck):
    """
    Summarize given dataset information based on pandas_profiling package.
    Can be used inside `Suite`
    """

    def run(self, dataset: Dataset, model=None) -> CheckResult:
        """
        Runs the dataset_info check

        Arguments:
            dataset: Dataset - The dataset object
            model: any = None - not used in the check

        Returns:
            the output of the dataset_info check
        """
        return dataset_info(dataset)

