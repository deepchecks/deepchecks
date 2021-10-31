"""Module contains model_info check."""
import pandas as pd
from mlchecks import CheckResult
from mlchecks.base import Dataset
from mlchecks.base.check import SingleDatasetBaseCheck

__all__ = ['columns_info', 'ColumnsInfo']


def columns_info(dataset:  Dataset):
    """Return type of each column.

    Args:
        dataset (Dataset): any dataset.

    Returns:
        CheckResult: value is diractory of a column and its type
    """
    dataset = Dataset.validate_dataset_or_dataframe(dataset)
    value = dataset.show_columns_types()
    df = pd.DataFrame.from_dict(value, orient='index', columns=['type']).transpose()

    return CheckResult(value, check=columns_info, display=df)


class ColumnsInfo(SingleDatasetBaseCheck):
    """Return type of each column."""

    def run(self, dataset: Dataset) -> CheckResult:
        """Run columns_info.

        Args:
          dataset (Dataset): any dataset.

        Returns:
          (CheckResult): value is diractory of a column and its type.
        """
        return columns_info(dataset)

