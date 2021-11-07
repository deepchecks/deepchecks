"""Module contains columns_info check."""
import pandas as pd
from mlchecks import CheckResult
from mlchecks.base import Dataset
from mlchecks.base.check import SingleDatasetBaseCheck

__all__ = ['ColumnsInfo']


class ColumnsInfo(SingleDatasetBaseCheck):
    """Return role of each column."""

    def run(self, dataset: Dataset) -> CheckResult:
        """Return the role and logical type of each column.

        Args:
          dataset (Dataset): any dataset.

        Returns:
          CheckResult: value is dictionary of a column and its role and logical type.
                       display a table of the dictionary.
        """
        return self._columns_info(dataset)

    def _columns_info(self, dataset: Dataset):
        dataset = Dataset.validate_dataset_or_dataframe(dataset)
        value = dataset.show_columns_info()
        df = pd.DataFrame.from_dict(value, orient='index', columns=['role']).transpose()

        return CheckResult(value, check=self.run, header='Columns Info', display=df)

