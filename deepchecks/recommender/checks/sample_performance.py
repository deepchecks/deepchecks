# ----------------------------------------------------------------------------
# Copyright (C) 2021-2023 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""Module containing the single dataset performance check for recommender system."""
from typing import TYPE_CHECKING, Callable, Dict, List, Mapping, Optional, TypeVar, Union
import plotly.express as px

import pandas as pd

from deepchecks.core import CheckResult, ConditionCategory, ConditionResult
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.recommender import Context
from deepchecks.tabular.base_checks import SingleDatasetCheck
from deepchecks.utils.strings import format_number
from deepchecks.utils.docref import doclink

if TYPE_CHECKING:
    from deepchecks.core.checks import CheckConfig

__all__ = ['SamplePerformance']

SDP = TypeVar('SDP', bound='SamplePerformance')


class SamplePerformance(SingleDatasetCheck):
    """Summarize given recommender system model performance based on selected scorers.

    Parameters
    ----------
    scorers: Union[Mapping[str, Union[str, Callable]], List[str]], default: None
        Scorers to override the default scorers, find more about the supported formats at
        https://docs.deepchecks.com/stable/user-guide/general/metrics_guide.html
    n_samples : int , default: 1_000_000
        number of samples to use for this check.

    k (int, optional), default: 20
        Top recommendations considered for certain metrics.

    random_state : int, default: 42
        random seed for all check internals.
    """

    def __init__(self,
                 scorers: Optional[Union[Mapping[str, Union[str, Callable]], List[str]]] = None,
                 n_samples: Union[int, None] = 1_000_000,
                 k : int = 10,
                 random_state: int = 42,
                 **kwargs):
        super().__init__(**kwargs)
        self.scorers = scorers
        self.n_samples = n_samples
        self.k = k
        self.random_state = random_state

    def run_logic(self, context: Context, dataset_kind) -> CheckResult:
        """Run check."""
        dataset = context.get_data_by_kind(dataset_kind).sample(self.n_samples,
                                                                random_state=self.random_state)
        model = context.model
        scorers = context.get_scorers(self.scorers, use_avg_defaults=True, k=self.k)
        results = []
        for scorer in scorers:
            scorer_value = scorer(model, dataset)
            results.append([f'{scorer.name}@{self.k}', scorer_value])
        results_df = pd.DataFrame(results, columns=['Metric', 'Value'])

        if context.with_display:
            fig = px.bar(results_df, y='Value', x='Metric')

        return CheckResult(results_df, header='Sample Performance', display=fig)

    def config(
        self,
        include_version: bool = True,
        include_defaults: bool = True
    ) -> 'CheckConfig':
        """Return check configuration."""
        if isinstance(self.scorers, dict):
            for k, v in self.scorers.items():
                if not isinstance(v, str):
                    reference = doclink(
                        'supported-metrics-by-string',
                        template='For a list of built-in scorers please refer to {link}'
                    )
                    raise ValueError(
                        'Only built-in scorers are allowed when serializing check instances. '
                        f'{reference}. Scorer name: {k}'
                    )
        return super().config(include_version=include_version, include_defaults=include_defaults)

    def reduce_output(self, check_result: CheckResult) -> Dict[str, float]:
        """Return the values of the metrics for the dataset provided in a {metric: value} format."""
        result = {}
        for _, row in check_result.value.iterrows():
            key = row['Metric'] if pd.isna(row.get('Class')) else (row['Metric'], str(row['Class']))
            result[key] = row['Value']
        return result

    def add_condition_greater_than(self, threshold: float, metrics: List[str] = None, class_mode: str = 'all') -> SDP:
        """Add condition - the selected metrics scores are greater than the threshold.

        Parameters
        ----------
        threshold: float
            The threshold that the metrics result should be grater than.
        metrics: List[str]
            The names of the metrics from the check to apply the condition to. If None, runs on all the metrics that
            were calculated in the check.
        class_mode: str, default: 'all'
            The decision rule over the classes, one of 'any', 'all', class name. If 'any', passes if at least one class
            result is above the threshold, if 'all' passes if all the class results are above the threshold,
            class name, passes if the result for this specified class is above the threshold.
        """

        def condition(check_result, metrics_to_check=metrics):
            metrics_to_check = check_result['Metric'].unique() if metrics_to_check is None else metrics_to_check
            metrics_pass = []

            for metric in metrics_to_check:
                if metric not in check_result.Metric.unique():
                    raise DeepchecksValueError(f'The requested metric was not calculated, the metrics calculated in '
                                               f'this check are: {check_result.Metric.unique()}.')
                metric_result = check_result[check_result['Metric'] == metric]
                if class_mode == 'all':
                    metrics_pass.append(min(metric_result['Value']) > threshold)
                elif class_mode == 'any':
                    metrics_pass.append(max(metric_result['Value']) > threshold)
                elif str(class_mode) in [str(x) for x in metric_result['Class'].unique()]:
                    metrics_pass.append(metric_result['Value'][class_mode] > threshold)
                else:
                    raise DeepchecksValueError(f'class_mode expected be one of the classes in the check results or any '
                                               f'or all, received {class_mode}.')

            if all(metrics_pass):
                return ConditionResult(ConditionCategory.PASS, 'Passed for all of the metrics.')
            else:
                failed_metrics = ([a for a, b in zip(metrics_to_check, metrics_pass) if not b])
                return ConditionResult(ConditionCategory.FAIL, f'Failed for metrics: {failed_metrics}')

        return self.add_condition(f'Selected metrics scores are greater than {format_number(threshold)}', condition)
