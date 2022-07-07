# ----------------------------------------------------------------------------
# Copyright (C) 2021-2022 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""Module contains columns_info check."""
import pandas as pd

from deepchecks.core import CheckResult
from deepchecks.tabular import Context, SingleDatasetCheck
from deepchecks.utils.features import N_TOP_MESSAGE, column_importance_sorter_dict

__all__ = ['ColumnsInfo']


class ColumnsInfo(SingleDatasetCheck):
    """Return the role and logical type of each column.

    Parameters
    ----------
    n_top_columns : int , optional
        amount of columns to show ordered by feature importance (date, index, label are first)
    """

    def __init__(self, n_top_columns: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.n_top_columns = n_top_columns

    def run_logic(self, context: Context, dataset_kind) -> CheckResult:
        """Run check.

        Returns
        -------
        CheckResult
            value is dictionary of a column and its role and logical type.
            display a table of the dictionary.
        """
        dataset = context.get_data_by_kind(dataset_kind)
        columns_info = dataset.columns_info
        columns_info = column_importance_sorter_dict(columns_info, dataset, context.feature_importance)

        columns_info_to_display = (
            columns_info
            if len(columns_info) <= self.n_top_columns
            else dict(list(columns_info.items())[:self.n_top_columns])
        )

        df = pd.DataFrame.from_dict(columns_info_to_display, orient='index', columns=['role'])
        df = df.transpose()

        return CheckResult(
            columns_info,
            header='Columns Info',
            display=[N_TOP_MESSAGE % self.n_top_columns, df]
        )
