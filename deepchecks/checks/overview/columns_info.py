# Deepchecks
# Copyright (C) 2021 Deepchecks
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
"""Module contains columns_info check."""
import pandas as pd
from deepchecks import CheckResult
from deepchecks.base import Dataset
from deepchecks.base.check import SingleDatasetBaseCheck
from deepchecks.utils.features import calculate_feature_importance_or_null, column_importance_sorter_dict


__all__ = ['ColumnsInfo']


class ColumnsInfo(SingleDatasetBaseCheck):
    """Return the role and logical type of each column.

    Args:
        n_top_columns (int): (optinal - used only if model was specified)
                             amount of columns to show ordered by feature importance (date, index, label are first)
    """

    def __init__(self, n_top_columns: int = 10):
        super().__init__()
        self.n_top_columns = n_top_columns

    def run(self, dataset: Dataset, model=None) -> CheckResult:
        """Run check.

        Args:
          dataset (Dataset): any dataset.

        Returns:
          CheckResult: value is dictionary of a column and its role and logical type.
          display a table of the dictionary.
        """
        feature_importances = calculate_feature_importance_or_null(dataset, model)
        return self._columns_info(dataset, feature_importances)

    def _columns_info(self, dataset: Dataset, feature_importances: pd.Series=None):
        dataset = Dataset.validate_dataset_or_dataframe(dataset)
        value = dataset.columns_info
        value = column_importance_sorter_dict(value, dataset, feature_importances, self.n_top_columns)
        df = pd.DataFrame.from_dict(value, orient='index', columns=['role'])
        df = df.transpose()

        return CheckResult(value, check=self.__class__, header='Columns Info', display=df)

