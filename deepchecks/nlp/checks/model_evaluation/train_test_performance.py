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
"""Module containing the Train-Test Performance check."""
import typing as t
from numbers import Number

import numpy as np
import pandas as pd
import plotly.express as px

from deepchecks.core import CheckResult
from deepchecks.nlp import Context, TrainTestCheck
from deepchecks.nlp.metric_utils.scorers import infer_on_text_data
from deepchecks.nlp.task_type import TaskType
from deepchecks.nlp.text_data import TextData
from deepchecks.utils.abstracts.train_test_performace import TrainTestPerformanceAbstract
from deepchecks.utils.plot import DEFAULT_DATASET_NAMES, colors

__all__ = ['TrainTestPerformance']


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

    def __init__(
        self,
        scorers: t.Union[
            t.Mapping[str, t.Union[str, t.Callable]],
            t.List[str],
            None
        ] = None,
        n_samples: int = 1_000_000,
        random_state: int = 42,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.scorers = scorers
        self.n_samples = n_samples
        self.random_state = random_state

    @classmethod
    def _default_per_class_scorers(cls) -> t.Dict[str, str]:
        # TODO:
        return {}

    def run_logic(self, context: Context) -> CheckResult:
        """Run check."""
        train_dataset = t.cast(TextData, context.train.sample(self.n_samples, random_state=self.random_state))
        test_dataset = t.cast(TextData, context.test.sample(self.n_samples, random_state=self.random_state))
        model = context.model
        scorers = context.get_scorers(self.scorers, use_avg_defaults=False)
        datasets = {'Train': train_dataset, 'Test': test_dataset}

        results = []

        for dataset_name, dataset in datasets.items():

            if context.task_type is TaskType.TEXT_CLASSIFICATION and dataset.is_multi_label_classification():
                n_samples_per_class = dict(enumerate(np.array(dataset.label).sum(axis=0)))
                n_of_labels = sum(n_samples_per_class.values())

            elif context.task_type is TaskType.TEXT_CLASSIFICATION:
                label = pd.Series(dataset.label)
                n_samples_per_class = label.groupby(label).count()
                n_of_labels = len(label)

            elif context.task_type is TaskType.TOKEN_CLASSIFICATION:
                # TODO:
                n_samples_per_class = {}
                n_of_labels = 0

            else:
                raise NotImplementedError()

            for scorer in scorers:
                scorer_value = infer_on_text_data(
                    scorer=scorer,
                    model=model,
                    data=dataset
                )
                if isinstance(scorer_value, Number):
                    results.append([
                        dataset_name,
                        pd.NA,
                        scorer.name,
                        scorer_value,
                        n_of_labels
                    ])
                else:
                    results.extend((
                        [
                            dataset_name,
                            class_name,
                            scorer.name,
                            class_score,
                            n_samples_per_class.get(class_name, 0)
                        ]
                        for class_name, class_score in scorer_value.items()
                    ))

        results_df = pd.DataFrame(
            results,
            columns=[
                'Dataset',
                'Class',
                'Metric',
                'Value',
                'Number of samples'
            ]
        )

        if context.with_display is False:
            figures = None
        else:
            display_df = results_df.replace({
                'Dataset': {
                    DEFAULT_DATASET_NAMES[0]: train_dataset.name or 'Train',
                    DEFAULT_DATASET_NAMES[1]: test_dataset.name or 'Test'
                }
            })

            figures = []
            data_scorers_per_class = display_df[results_df['Class'].notna()]
            data_scorers_per_dataset = display_df[results_df['Class'].isna()].drop(columns=['Class'])

            for data in (data_scorers_per_dataset, data_scorers_per_class):
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
                    color_discrete_map={
                        train_dataset.name: colors[DEFAULT_DATASET_NAMES[0]],
                        test_dataset.name: colors[DEFAULT_DATASET_NAMES[1]]
                    },
                )

                if 'Class' in data.columns:
                    fig.update_xaxes(tickprefix='Class ', tickangle=60)

                figures.append(
                    fig.update_xaxes(title=None, type='category')
                    .update_yaxes(title=None, matches=None)
                    .for_each_annotation(lambda a: a.update(text=a.text.split('=')[-1]))
                    .for_each_yaxis(lambda yaxis: yaxis.update(showticklabels=True))
                )

        return CheckResult(
            results_df,
            header='Train Test Performance',
            display=figures
        )
