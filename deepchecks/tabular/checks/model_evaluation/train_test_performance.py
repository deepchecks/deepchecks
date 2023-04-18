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
"""Module containing the train test performance check."""
from numbers import Number
from typing import Callable, Dict, List, Mapping, TypeVar, Union, cast

import pandas as pd
import plotly.express as px

from deepchecks.core import CheckResult
from deepchecks.core.checks import CheckConfig
from deepchecks.tabular import Context, TrainTestCheck
from deepchecks.tabular.metric_utils import MULTICLASS_SCORERS_NON_AVERAGE
from deepchecks.utils.abstracts.train_test_performace import TrainTestPerformanceAbstract
from deepchecks.utils.docref import doclink
from deepchecks.utils.plot import DEFAULT_DATASET_NAMES, colors

__all__ = ['TrainTestPerformance']


PR = TypeVar('PR', bound='TrainTestPerformance')


class TrainTestPerformance(TrainTestPerformanceAbstract, TrainTestCheck):
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

    Notes
    -----
    Scorers are a convention of sklearn to evaluate a model.
    `See scorers documentation <https://scikit-learn.org/stable/modules/model_evaluation.html#scoring>`__
    A scorer is a function which accepts (model, X, y_true) and returns a float result which is the score.
    For every scorer higher scores are better than lower scores.

    You can create a scorer out of existing sklearn metrics:

    .. code-block:: python

        from sklearn.metrics import roc_auc_score, make_scorer

        training_labels = [1, 2, 3]
        auc_scorer = make_scorer(roc_auc_score, labels=training_labels, multi_class='ovr')
        # Note that the labels parameter is required for multi-class classification in metrics like roc_auc_score or
        # log_loss that use the predict_proba function of the model, in case that not all labels are present in the test
        # set.

    Or you can implement your own:

    .. code-block:: python

        from sklearn.metrics import make_scorer


        def my_mse(y_true, y_pred):
            return (y_true - y_pred) ** 2


        # Mark greater_is_better=False, since scorers always suppose to return
        # value to maximize.
        my_mse_scorer = make_scorer(my_mse, greater_is_better=False)
    """

    def __init__(self,
                 scorers: Union[Mapping[str, Union[str, Callable]], List[str]] = None,
                 n_samples: int = 1_000_000,
                 random_state: int = 42,
                 **kwargs):
        super().__init__(**kwargs)
        self.scorers = scorers
        self.n_samples = n_samples
        self.random_state = random_state

    @classmethod
    def _default_per_class_scorers(cls) -> Dict[str, str]:
        return MULTICLASS_SCORERS_NON_AVERAGE

    def run_logic(self, context: Context) -> CheckResult:
        """Run check."""
        train_dataset = context.train.sample(self.n_samples, random_state=self.random_state)
        test_dataset = context.test.sample(self.n_samples, random_state=self.random_state)
        model = context.model
        scorers = context.get_scorers(self.scorers, use_avg_defaults=False)
        datasets = {'Train': train_dataset, 'Test': test_dataset}

        results = []
        for dataset_name, dataset in datasets.items():
            label = cast(pd.Series, dataset.label_col)
            n_samples_per_class = label.groupby(label).count()
            for scorer in scorers:
                scorer_value = scorer(model, dataset)
                if isinstance(scorer_value, Number):
                    results.append([dataset_name, pd.NA, scorer.name, scorer_value, len(label)])
                else:
                    results.extend(
                        [[dataset_name, class_name, scorer.name, class_score, n_samples_per_class.get(class_name, 0)]
                            for class_name, class_score in scorer_value.items()])

        results_df = pd.DataFrame(results, columns=['Dataset', 'Class', 'Metric', 'Value', 'Number of samples'])

        if context.with_display:
            results_df_for_display = results_df.copy()
            results_df_for_display['Dataset']\
                .replace({DEFAULT_DATASET_NAMES[0]: train_dataset.name, DEFAULT_DATASET_NAMES[1]: test_dataset.name},
                         inplace=True)
            figs = []
            data_scorers_per_class = results_df_for_display[results_df['Class'].notna()]
            data_scorers_per_dataset = results_df_for_display[results_df['Class'].isna()].drop(columns=['Class'])
            for data in [data_scorers_per_dataset, data_scorers_per_class]:
                if data.shape[0] == 0:
                    continue
                fig = px.histogram(
                    data,
                    x='Class' if 'Class' in data.columns else 'Dataset',
                    y='Value',
                    color='Dataset',
                    barmode='group',
                    facet_col='Metric',
                    facet_col_spacing=0.05,
                    hover_data=['Number of samples'],
                    color_discrete_map={train_dataset.name: colors[DEFAULT_DATASET_NAMES[0]],
                                        test_dataset.name: colors[DEFAULT_DATASET_NAMES[1]]},
                )
                if 'Class' in data.columns:
                    fig.update_xaxes(tickprefix='Class ', tickangle=60)
                fig = (
                    fig.update_xaxes(title=None, type='category')
                    .update_yaxes(title=None, matches=None)
                    .for_each_annotation(lambda a: a.update(text=a.text.split('=')[-1]))
                    .for_each_yaxis(lambda yaxis: yaxis.update(showticklabels=True)))
                figs.append(fig)
        else:
            figs = None

        return CheckResult(
            results_df,
            header='Train Test Performance',
            display=figs
        )

    def config(self, include_version: bool = True, include_defaults: bool = True) -> CheckConfig:
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
