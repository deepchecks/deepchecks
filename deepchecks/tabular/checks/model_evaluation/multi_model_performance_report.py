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
"""Module containing multi model performance report check."""
import warnings
from typing import TYPE_CHECKING, Callable, Dict, List, Mapping, Union, cast

import pandas as pd
import plotly.express as px

from deepchecks.core import CheckResult
from deepchecks.tabular import ModelComparisonCheck, ModelComparisonContext
from deepchecks.tabular.utils.task_type import TaskType
from deepchecks.utils.docref import doclink

if TYPE_CHECKING:
    from deepchecks.core.checks import CheckConfig

__all__ = ['MultiModelPerformanceReport']


class MultiModelPerformanceReport(ModelComparisonCheck):
    """Summarize performance scores for multiple models on test datasets.

    Parameters
    ----------
    scorers: Union[Mapping[str, Union[str, Callable]], List[str]], default: None
        Scorers to override the default scorers, find more about the supported formats at
        https://docs.deepchecks.com/stable/user-guide/general/metrics_guide.html
    alternative_scorers : Dict[str, Callable] , default: None
        Deprecated, please use scorers instead.
    n_samples : int , default: 1_000_000
        number of samples to use for this check.
    random_state : int, default: 42
        random seed for all check internals.
    """

    def __init__(self,
                 scorers: Union[Mapping[str, Union[str, Callable]], List[str]] = None,
                 alternative_scorers: Dict[str, Callable] = None,
                 n_samples: int = 1_000_000,
                 random_state: int = 42,
                 **kwargs):
        super().__init__(**kwargs)
        if alternative_scorers is not None:
            warnings.warn(f'{self.__class__.__name__}: alternative_scorers is deprecated. Please use scorers instead.',
                          DeprecationWarning)
            self.scorers = alternative_scorers
        else:
            self.alternative_scorers = scorers
        self.n_samples = n_samples
        self.random_state = random_state

    def run_logic(self, multi_context: ModelComparisonContext):
        """Run check logic."""
        first_context = multi_context[0]
        scorers = first_context.get_scorers(self.alternative_scorers, use_avg_defaults=False)

        if multi_context.task_type in [TaskType.MULTICLASS, TaskType.BINARY]:
            plot_x_axis = ['Class', 'Model']
            results = []

            for context, model_name in zip(multi_context, multi_context.models.keys()):
                test = context.test.sample(self.n_samples, random_state=self.random_state)
                model = context.model
                label = cast(pd.Series, test.label_col)
                n_samples = label.groupby(label).count()
                results.extend(
                    [model_name, class_score, scorer.name, class_name, n_samples[class_name]]
                    for scorer in scorers
                    # scorer returns numpy array of results with item per class
                    for class_score, class_name in zip(scorer(model, test), context.model_classes)
                )

            results_df = pd.DataFrame(results, columns=['Model', 'Value', 'Metric', 'Class', 'Number of samples'])

        else:
            plot_x_axis = 'Model'
            results = [
                [model_name, scorer(context.model, context.test), scorer.name,
                 cast(pd.Series, context.test.label_col).count()]
                for context, model_name in zip(multi_context, multi_context.models.keys())
                for scorer in scorers
            ]
            results_df = pd.DataFrame(results, columns=['Model', 'Value', 'Metric', 'Number of samples'])

        fig = px.histogram(
            results_df,
            x=plot_x_axis,
            y='Value',
            color='Model',
            barmode='group',
            facet_col='Metric',
            facet_col_spacing=0.05,
            hover_data=['Number of samples'],
        )

        if multi_context.task_type in [TaskType.MULTICLASS, TaskType.BINARY]:
            fig.update_xaxes(title=None, tickprefix='Class ', tickangle=60)
        else:
            fig.update_xaxes(title=None)

        fig = (
            fig.update_yaxes(title=None, matches=None)
            .for_each_annotation(lambda a: a.update(text=a.text.split('=')[-1]))
            .for_each_yaxis(lambda yaxis: yaxis.update(showticklabels=True))
        )

        return CheckResult(results_df, display=[fig])

    def config(self, include_version: bool = True, include_defaults: bool = True) -> 'CheckConfig':
        """Return check instance config."""
        if self.alternative_scorers is not None:
            for k, v in self.alternative_scorers.items():
                if not isinstance(v, str):
                    reference = doclink(
                        'supported-metrics-by-string',
                        template='For a list of built-in scorers please refer to {link}. '
                    )
                    raise ValueError(
                        'Only built-in scorers are allowed when serializing check instances. '
                        f'{reference}Scorer name: {k}'
                    )
        return super().config(include_version, include_defaults=include_defaults)
