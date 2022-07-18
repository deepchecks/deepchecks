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
"""Module containing the train test performance check."""
from numbers import Number
from typing import Callable, Dict, List, TypeVar, Union, cast

import pandas as pd
import plotly.express as px

from deepchecks.core import CheckResult
from deepchecks.core.check_utils.class_performance_utils import (
    get_condition_class_performance_imbalance_ratio_less_than, get_condition_test_performance_greater_than,
    get_condition_train_test_relative_degradation_less_than)
from deepchecks.core.checks import DatasetKind, ReduceMixin
from deepchecks.tabular import Context, TrainTestCheck
from deepchecks.tabular.metric_utils import MULTICLASS_SCORERS_NON_AVERAGE
from deepchecks.utils.plot import colors
from deepchecks.utils.strings import format_percent

__all__ = ['TrainTestPerformance']


PR = TypeVar('PR', bound='TrainTestPerformance')


class TrainTestPerformance(TrainTestCheck, ReduceMixin):
    """Summarize given model performance on the train and test datasets based on selected scorers.

    Parameters
    ----------
    scorers : Union[List[str], Dict[str, Union[str, Callable]]], default: None
        List of scorers to use. If None, use default scorers.
        Scorers can be supplied as a list of scorer names or as a dictionary of names and functions.
    reduce: Union[Callable, str], default: 'mean'
        An optional argument only used for the reduce_output function when using
        per-class scorers.

    Notes
    -----
    Scorers are a convention of sklearn to evaluate a model.
    `See scorers documentation <https://scikit-learn.org/stable/modules/model_evaluation.html#scoring>`_
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
                 scorers: Union[List[str], Dict[str, Union[str, Callable]]] = None,
                 reduce: Union[Callable, str] = 'mean',
                 **kwargs):
        super().__init__(**kwargs)
        self.user_scorers = scorers
        self.reduce = reduce

    def run_logic(self, context: Context) -> CheckResult:
        """Run check."""
        train_dataset = context.train
        test_dataset = context.test
        model = context.model
        scorers = context.get_scorers(self.user_scorers, use_avg_defaults=False)
        datasets = {'Train': train_dataset, 'Test': test_dataset}

        results = []
        for dataset_name, dataset in datasets.items():
            classes = dataset.classes
            label = cast(pd.Series, dataset.label_col)
            n_samples = label.groupby(label).count()
            for scorer in scorers:
                scorer_value = scorer(model, dataset)
                if isinstance(scorer_value, Number):
                    results.append([dataset_name, None, scorer.name, scorer_value, len(label)])
                else:
                    results.extend(
                        [[dataset_name, class_name, scorer.name, class_score, n_samples[class_name]]
                            for class_score, class_name in zip(scorer_value, classes)])

        results_df = pd.DataFrame(results, columns=['Dataset', 'Class', 'Metric', 'Value', 'Number of samples'])

        if context.with_display:
            figs = []
            data_scorers_per_class = results_df[results_df['Class'].notna()]
            data_scorers_per_dataset = results_df[results_df['Class'].isna()].drop(columns=['Class'])
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
                    color_discrete_map={'Train': colors['Train'], 'Test': colors['Test']},
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

    def reduce_output(self, check_result: CheckResult) -> Dict[str, float]:
        """Return the values of the metrics for the test dataset in {metric: value} format."""
        df = check_result.value
        df = df[df['Dataset'] == DatasetKind.TEST.value]
        df = df.groupby('Metric').aggregate(self.reduce).reset_index()
        return dict(zip(df['Metric'], df['Value']))

    def add_condition_test_performance_greater_than(self: PR, min_score: float) -> PR:
        """Add condition - metric scores are greater than the threshold.

        Parameters
        ----------
        min_score : float
            Minimum score to pass the check.
        """
        condition = get_condition_test_performance_greater_than(min_score=min_score)

        return self.add_condition(f'Scores are greater than {min_score}', condition)

    def add_condition_train_test_relative_degradation_less_than(self: PR, threshold: float = 0.1) -> PR:
        """Add condition - test performance is not degraded by more than given percentage in train.

        Parameters
        ----------
        threshold : float , default: 0.1
            maximum degradation ratio allowed (value between 0 and 1)
        """
        condition = get_condition_train_test_relative_degradation_less_than(threshold=threshold)

        return self.add_condition(f'Train-Test scores relative degradation is less than {threshold}',
                                  condition)

    def add_condition_class_performance_imbalance_ratio_less_than(
        self: PR,
        threshold: float = 0.3,
        score: str = None
    ) -> PR:
        """Add condition - relative ratio difference between highest-class and lowest-class is less than threshold.

        Parameters
        ----------
        threshold : float , default: 0.3
            ratio difference threshold
        score : str , default: None
            limit score for condition

        Returns
        -------
        Self
            instance of 'ClassPerformance' or it subtype

        Raises
        ------
        DeepchecksValueError
            if unknown score function name were passed.
        """
        if score is None:
            score = next(iter(MULTICLASS_SCORERS_NON_AVERAGE))

        condition = get_condition_class_performance_imbalance_ratio_less_than(threshold=threshold, score=score)

        return self.add_condition(
            name=f'Relative ratio difference between labels \'{score}\' score is less than {format_percent(threshold)}',
            condition_func=condition
        )
