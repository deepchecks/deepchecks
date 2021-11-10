"""Module contains columns_info check."""
import pandas as pd
from mlchecks import CheckResult
from mlchecks.base import Dataset
from mlchecks.base.check import SingleDatasetBaseCheck
from mlchecks.feature_importance_utils import calculate_feature_importance, column_importance_sorter_df
from mlchecks.utils import MLChecksValueError
__all__ = ['ColumnsInfo']


class ColumnsInfo(SingleDatasetBaseCheck):
    """Return the role and logical type of each column."""

    def __init__(self, n_top: int=10):
        super().__init__()
        self.n_top = n_top
     
    def run(self, dataset: Dataset, model=None) -> CheckResult:
        """Run check.

        Args:
          dataset (Dataset): any dataset.

        Returns:
          CheckResult: value is dictionary of a column and its role and logical type.
                       display a table of the dictionary.
        """
        feature_importances = None
        if model:
          try:
            feature_importances = calculate_feature_importance(dataset, model)
          except MLChecksValueError:
            pass
        return self._columns_info(dataset, feature_importances)

    def _columns_info(self, dataset: Dataset, feature_importances: pd.Series=None):
        dataset = Dataset.validate_dataset_or_dataframe(dataset)
        value = dataset.show_columns_info()
        df = pd.DataFrame.from_dict(value, orient='index', columns=['role'])
        df = column_importance_sorter_df(df, dataset, feature_importances, self.n_top)
        df = df.transpose()

        return CheckResult(value, check=self.__class__, header='Columns Info', display=df)

