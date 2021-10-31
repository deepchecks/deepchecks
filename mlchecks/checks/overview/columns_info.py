"""Module contains model_info check."""
import pandas as pd
from mlchecks import CheckResult
from mlchecks.base import Dataset
from mlchecks.base.check import SingleDatasetBaseCheck

__all__ = ['columns_info', 'ColumnsInfo']


def columns_info(dataset:  Dataset):
    """Return role of each column.

    Args:
        dataset (Dataset): any dataset.

    Returns:
        CheckResult: value is diractory of a column and its role
    """
    dataset = Dataset.validate_dataset_or_dataframe(dataset)
    value = dataset.show_columns_roles()
    df = pd.DataFrame.from_dict(value, orient='index', columns=['role']).transpose()

    return CheckResult(value, check=columns_info, display=df)


class ColumnsInfo(SingleDatasetBaseCheck):
    """Return role of each column."""

    def run(self, dataset: Dataset) -> CheckResult:
        """Run columns_info.

        Args:
          dataset (Dataset): any dataset.

        Returns:
          (CheckResult): value is diractory of a column and its role.
        """
        return columns_info(dataset)

