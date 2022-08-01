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
from typing import Callable, Dict, cast

import pandas as pd
import plotly.express as px

from deepchecks.core import CheckResult
from deepchecks.tabular import ModelComparisonCheck, ModelComparisonContext
from deepchecks.tabular.utils.task_type import TaskType

__all__ = ['MultiModelPerformanceReport']


class MultiModelPerformanceReport(ModelComparisonCheck):
    """Summarize performance scores for multiple models on test datasets.

    Parameters
    ----------
    alternative_scorers : Dict[str, Callable] , default: None
        An optional dictionary of scorer name to scorer functions.
        If none given, using default scorers
    """

    def __init__(self, alternative_scorers: Dict[str, Callable] = None, **kwargs):
        super().__init__(**kwargs)
        self.user_scorers = alternative_scorers

    def run_logic(self, multi_context: ModelComparisonContext):
        """Run check logic."""
        first_context = multi_context[0]
        scorers = first_context.get_scorers(self.user_scorers, use_avg_defaults=False)

        if multi_context.task_type in [TaskType.MULTICLASS, TaskType.BINARY]:
            plot_x_axis = ['Class', 'Model']
            results = []

            for context, model_name in zip(multi_context, multi_context.models.keys()):
                test = context.test
                model = context.model
                label = cast(pd.Series, test.label_col)
                n_samples = label.groupby(label).count()
                results.extend(
                    [model_name, class_score, scorer.name, class_name, n_samples[class_name]]
                    for scorer in scorers
                    # scorer returns numpy array of results with item per class
                    for class_score, class_name in zip(scorer(model, test), test.classes)
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
