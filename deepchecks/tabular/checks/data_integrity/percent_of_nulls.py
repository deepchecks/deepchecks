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

import pandas as pd
import plotly.express as px

from deepchecks.core.check_result import CheckResult
from deepchecks.core.checks import DatasetKind
from deepchecks.core.condition import ConditionCategory, ConditionResult
from deepchecks.core.reduce_classes import ReduceFeatureMixin
from deepchecks.tabular import Context, SingleDatasetCheck
from deepchecks.tabular._shared_docs import docstrings
from deepchecks.tabular.utils.messages import get_condition_passed_message
from deepchecks.utils.strings import format_percent
from deepchecks.utils.typing import Hashable

__all__ = ['PercentOfNulls']

TPercentOfNulls = t.TypeVar('TPercentOfNulls', bound='PercentOfNulls')


@docstrings
class PercentOfNulls(SingleDatasetCheck, ReduceFeatureMixin):
    """Percent of 'Null' values in each column.

    Parameters
    ----------
    columns : Union[Hashable, List[Hashable]] , default: None
        List of columns to check, if none given checks
        all columns Except ignored ones.
    ignore_columns : Union[Hashable, List[Hashable]] , default: None
        List of columns to ignore, if none given checks
        based on columns variable.
    max_features_to_show : int , default: 5
        maximum features with to show, showing top features based on percent of nulls.
    aggregation_method: str, default: 'max'
        {feature_aggregation_method_argument:2*indent}
    n_samples : int , default: 100_000
        number of samples to use for this check.
    random_state : int, default: 42
        random seed for all check internals.
    """

    def __init__(
            self,
            columns: t.Union[Hashable, t.List[Hashable], None] = None,
            ignore_columns: t.Union[Hashable, t.List[Hashable], None] = None,
            max_features_to_show: int = 5,
            aggregation_method='max',
            n_samples: int = 100_000,
            random_state: int = 42,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.columns = columns
        self.ignore_columns = ignore_columns
        self.max_features_to_show = max_features_to_show
        self.aggregation_method = aggregation_method
        self.n_samples = n_samples
        self.random_state = random_state

    def run_logic(self, context: Context, dataset_kind: DatasetKind) -> CheckResult:
        """Run check logic."""
        dataset = context.get_data_by_kind(dataset_kind).sample(self.n_samples, random_state=self.random_state)
        dataset = dataset.select(self.columns, self.ignore_columns, keep_label=False)
        data = dataset.features_columns

        feature_importance = context.feature_importance if context.feature_importance is not None \
            else pd.Series(index=list(data.columns), dtype=object)

        result_data = [[col, data[col].isna().sum(), feature_importance[col]] for col in data.columns]
        result_data = pd.DataFrame(data=result_data,
                                   columns=['Column',
                                            'Percent of nulls in sample',
                                            'Feature importance']).set_index(['Column'])
        result_data['Percent of nulls in sample'] = result_data['Percent of nulls in sample'] / dataset.n_samples
        result_data.sort_values(by='Percent of nulls in sample')
        if all(feature_importance.isna()):
            result_data.drop('Feature importance', axis=1, inplace=True)

        if context.with_display and max(result_data['Percent of nulls in sample']) > 0:
            display = (
                [px.bar(x=data.columns, y=result_data['Percent of nulls in sample'],
                        title='Percent Of Nulls', range_y=(0, 1))
                 .update_layout(yaxis_title=None, xaxis_title=None)])
        else:
            display = None

        return CheckResult(
            value=result_data,
            display=display,
            header='PercentOfNulls'
        )

    def reduce_output(self, check_result: CheckResult) -> t.Dict[str, float]:
        """Return an aggregated drift score based on aggregation method defined."""
        feature_importance = check_result.value['Feature importance'] if 'Feature importance' \
                                                                         in check_result.value.columns else None
        values = check_result.value['Percent of nulls in sample']
        return self.feature_reduce(self.aggregation_method, values, feature_importance, 'Null Ratio')

    def add_condition_percent_of_nulls_not_greater_than(self, threshold: float = 0.05) -> TPercentOfNulls:
        """Add condition - percent of null values in each column is not greater than the threshold.

        Parameters
        ----------
        threshold : float , default: 0.05
            Maximum threshold allowed.
        """

        def condition(result: pd.DataFrame) -> ConditionResult:
            failing = result[result['Percent of nulls in sample'] > threshold][
                'Percent of nulls in sample'].apply(format_percent)
            if len(failing) > 0:
                return ConditionResult(ConditionCategory.FAIL,
                                       f'Found {len(failing)} columns with ratio '
                                       f'of nulls above threshold: \n{dict(failing)}')
            else:
                details = get_condition_passed_message(len(result))
                if any(result['Percent of nulls in sample'] > 0):
                    features_with_null = result[result['Percent of nulls in sample'] > 0]
                    value_for_print = dict(features_with_null['Percent of nulls in sample'].apply(format_percent)[:5])
                    details += f'. Top columns with null ratio: \n{value_for_print}'
                return ConditionResult(ConditionCategory.PASS, details)

        return self.add_condition(
            f'Percent of null values in each column is not greater than {format_percent(threshold)}', condition)
