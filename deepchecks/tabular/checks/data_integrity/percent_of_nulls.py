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
"""Module with 'PercentOfNulls' check."""
import typing as t

import plotly.express as px

from deepchecks.core.check_result import CheckResult
from deepchecks.core.checks import DatasetKind, ReduceMixin
from deepchecks.core.condition import ConditionCategory, ConditionResult
from deepchecks.tabular import Context, SingleDatasetCheck
from deepchecks.utils.strings import format_list, format_percent
from deepchecks.utils.typing import Hashable

__all__ = ['PercentOfNulls']


TPercentOfNulls = t.TypeVar('TPercentOfNulls', bound='PercentOfNulls')


class PercentOfNulls(SingleDatasetCheck, ReduceMixin):
    """Percent of 'Null' values in each column.

    Parameters
    ----------
    columns : Union[Hashable, List[Hashable]] , default: None
        List of columns to check, if none given checks
        all columns Except ignored ones.
    ignore_columns : Union[Hashable, List[Hashable]] , default: None
        List of columns to ignore, if none given checks
        based on columns variable.
    """

    def __init__(
        self,
        columns: t.Union[Hashable, t.List[Hashable], None] = None,
        ignore_columns: t.Union[Hashable, t.List[Hashable], None] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.columns = columns
        self.ignore_columns = ignore_columns

    def run_logic(self, context: Context, dataset_kind: DatasetKind) -> CheckResult:
        """Run check logic."""
        dataset = context.get_data_by_kind(dataset_kind)
        dataset = dataset.select(self.columns, self.ignore_columns, keep_label=False)
        data = dataset.features_columns

        l = len(data)
        n_of_nulls = [data[c].isna().sum() for c in data.columns]
        percent_of_nulls = [it / l for it in n_of_nulls]

        display = (
            [px.bar(x=data.columns, y=percent_of_nulls, title='Percent Of Nulls', range_y=(0, 1))]
            if sum(n_of_nulls) > 0 and context.with_display is True
            else None
        )
        return CheckResult(
            value=dict(zip(data.columns, percent_of_nulls)),
            display=display,
            header='PercentOfNulls'
        )

    def reduce_output(self, check_result: CheckResult) -> t.Dict[str, float]:
        """Reduce check result value.

        Returns
        -------
        Dict[str, float]
            percent of nulls in each columns
        """
        return check_result.value

    def add_condition_percent_of_nulls_not_greater_than(
        self: TPercentOfNulls,
        threshold: float = 0.05
    ) -> TPercentOfNulls:
        """Add condition - percent of null values in each column is not greater than the threshold.

        Parameters
        ----------
        threshold : float , default: 0.05
            Maximum threshold allowed.
        """
        def condition(result: t.Dict[str, float]) -> ConditionResult:
            columns = [k for k, v in result.items() if v > threshold]
            if len(columns) == 0:
                category = ConditionCategory.PASS
                details = ''
            else:
                category = ConditionCategory.FAIL
                details = (
                    'Columns with percent of null values greater than '
                    f'{format_percent(threshold)}: {format_list(columns)}'
                )
            return ConditionResult(category, details)

        return self.add_condition(
            f'Percent of null values in each column is not greater than {format_percent(threshold)}',
            condition
        )
