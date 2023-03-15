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
"""Module containing the single dataset performance check."""
from numbers import Number
from typing import TYPE_CHECKING, Callable, Dict, List, Mapping, Optional, TypeVar, Union, cast

import pandas as pd
import plotly.graph_objects as go

from deepchecks.core import CheckResult, ConditionCategory, ConditionResult
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.core.reduce_classes import ReduceMetricClassMixin
from deepchecks.tabular import Context
from deepchecks.tabular.base_checks import SingleDatasetCheck
from deepchecks.tabular.utils.task_type import TaskType
from deepchecks.utils.docref import doclink
from deepchecks.utils.strings import format_number

if TYPE_CHECKING:
    from deepchecks.core.checks import CheckConfig

__all__ = ['SamplePerformance']


SDP = TypeVar('SDP', bound='SamplePerformance')


class SamplePerformance(SingleDatasetCheck):
    """Summarize given model performance on the train and test datasets based on selected scorers.

    Parameters
    ----------
    scorers: Union[Mapping[str, Union[str, Callable]], List[str]], default: None
        Scorers to override the default scorers, find more about the supported formats at
        https://docs.deepchecks.com/stable/user-guide/general/metrics_guide.html
    n_samples : int , default: 1_000_000
        number of samples to use for this check.
    random_state : int, default: 42
        random seed for all check internals.
    """

    def __init__(self,
                 scorers: Optional[Union[Mapping[str, Union[str, Callable]], List[str]]] = None,
                 n_samples: int = 1_000_000,
                 random_state: int = 42,
                 **kwargs):
        super().__init__(**kwargs)
        self.scorers = scorers
        self.n_samples = n_samples
        self.random_state = random_state

    def run_logic(self, context: Context, dataset_kind) -> CheckResult:
        """Run check."""
        dataset = context.get_data_by_kind(dataset_kind).sample(self.n_samples, random_state=self.random_state)
        model = context.model
        scorers = context.get_scorers(self.scorers, use_avg_defaults=False)

        results = []
        display = None
        for scorer in scorers:
            scorer_value = scorer(model, dataset)
            results.append([scorer.name, scorer_value])
        results_df = pd.DataFrame(results, columns=['Metric', 'Value'])
        if context.with_display:
            fig = go.Figure()
            for _, row in results_df.iterrows():
                box_fig = go.Box(y=row['Value'], name=row['Metric'])
                fig.add_trace(box_fig)
            display = fig

        return CheckResult(results_df, display=display)

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
