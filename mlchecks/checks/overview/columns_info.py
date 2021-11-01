"""Module contains columns_info check."""
import pandas as pd
from mlchecks import CheckResult
from mlchecks.base import Dataset
from mlchecks.base.check import SingleDatasetBaseCheck

__all__ = ['columns_info', 'ColumnsInfo']


def columns_info(dataset:  Dataset):
    """Return the role and logical type of each column.

    Args:
        dataset (Dataset): any dataset.

    Returns:
        CheckResult: value is dictionary of a column and its role and logical type.
                     display a table of the dictionary.
    """
    dataset = Dataset.validate_dataset_or_dataframe(dataset)
    value = dataset.show_columns_info()
    df = pd.DataFrame.from_dict(value, orient='index', columns=['role']).transpose()

    return CheckResult(value, check=columns_info, display=df)


class ColumnsInfo(SingleDatasetBaseCheck):
    """Return role of each column."""

    def run(self, dataset: Dataset) -> CheckResult:
        """Run columns_info.

        Args:
          dataset (Dataset): any dataset.

        Returns:
          CheckResult: value is dictionary of a column and its role and logical type.
                       display a table of the dictionary.
        """
        return columns_info(dataset)

