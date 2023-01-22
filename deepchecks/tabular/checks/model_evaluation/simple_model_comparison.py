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
"""Module containing simple comparison check."""
import warnings
from collections import defaultdict
from numbers import Number
from typing import TYPE_CHECKING, Callable, Dict, Hashable, List, Mapping, Union

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from deepchecks.core import CheckResult, ConditionResult
from deepchecks.core.condition import ConditionCategory
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.tabular import Context, Dataset, TrainTestCheck
from deepchecks.tabular.utils.task_type import TaskType
from deepchecks.utils.distribution.preprocessing import ScaledNumerics
from deepchecks.utils.docref import doclink
from deepchecks.utils.metrics import get_gain
from deepchecks.utils.simple_models import ClassificationUniformModel, RandomModel, RegressionUniformModel
from deepchecks.utils.strings import format_percent

if TYPE_CHECKING:
    from deepchecks.core.checks import CheckConfig

__all__ = ['SimpleModelComparison']

_allowed_strategies = (
    'stratified',
    'most_frequent',
    'uniform',
    'tree'
)


class SimpleModelComparison(TrainTestCheck):
    """Compare given model score to simple model score (according to given model type).

    Parameters
    ----------
    strategy : str, default: 'most_frequent'
        Strategy to use to generate the predictions of the simple model ['stratified', 'uniform',
        'most_frequent', 'tree'].

        * `stratified`: randomly draw a label based on the train set label distribution. (Previously 'random')
        * `uniform`: in regression samples predictions uniformly at random from the y ranges. in classification draws
           predictions uniformly at random from the list of values in y.
        * `most_frequent`: in regression is mean value, in classification the most common value. (Previously 'constant')
        * `tree`: runs a simple decision tree.
    scorers: Union[Mapping[str, Union[str, Callable]], List[str]], default: None
        Scorers to override the default scorers, find more about the supported formats at
        https://docs.deepchecks.com/stable/user-guide/general/metrics_guide.html
    alternative_scorers : Dict[str, Callable], default: None
        Deprecated, please use scorers instead.
    max_gain : float , default: 50
        the maximum value for the gain value, limits from both sides [-max_gain, max_gain]
    max_depth : int , default: 3
        the max depth of the tree (used only if simple model type is tree).
    n_samples : int , default: 1_000_000
        number of samples to use for this check.
    random_state : int , default: 42
        the random state (used only if simple model type is tree or random).

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
        strategy: str = 'most_frequent',
        scorers: Union[Mapping[str, Union[str, Callable]], List[str]] = None,
        alternative_scorers: Dict[str, Callable] = None,
        max_gain: float = 50,
        max_depth: int = 3,
        n_samples: int = 1_000_000,
        random_state: int = 42,
        **kwargs
    ):
        super().__init__(**kwargs)
        if alternative_scorers is not None:
            warnings.warn(f'{self.__class__.__name__}: alternative_scorers is deprecated. Please use scorers instead.',
                          DeprecationWarning)
            self.alternative_scorers = alternative_scorers
        else:
            self.alternative_scorers = scorers
        self.max_gain = max_gain
        self.max_depth = max_depth
        self.n_samples = n_samples
        self.random_state = random_state
        self.strategy = strategy

        if self.strategy not in _allowed_strategies:
            raise DeepchecksValueError(
                f'{self.__class__.__name__}: strategy {self.strategy} is not allowed. '
                f'allowed strategies are {_allowed_strategies}.'
            )

    def run_logic(self, context: Context) -> CheckResult:
        """Run check.

        Returns
        -------
        CheckResult
            value is a Dict of: given_model_score, simple_model_score, ratio <br>
            ratio is given model / simple model (if the scorer returns negative values we divide 1 by it) <br>
            if ratio is infinite max_ratio is returned

        Raises
        ------
        DeepchecksValueError
            If the object is not a Dataset instance.
        """
        train_dataset = context.train.sample(self.n_samples, random_state=self.random_state)
        test_dataset = context.test.sample(self.n_samples, random_state=self.random_state)
        test_label = test_dataset.label_col
        task_type = context.task_type
        model = context.model

        # If user defined scorers used them, else use a single scorer
        if self.alternative_scorers:
            scorers = context.get_scorers(self.alternative_scorers, use_avg_defaults=False)
        else:
            scorers = [context.get_single_scorer(use_avg_defaults=False)]

        simple_model = self._create_simple_model(train_dataset, task_type)

        models = [
            (f'{type(model).__name__} model', 'Origin', model),
            (f'Simple model - {self.strategy}', 'Simple', simple_model)
        ]
        classes_display_array = []
        display_array = []
        # Multiclass have different return type from the scorer, list of score per class instead of single score
        if task_type in [TaskType.MULTICLASS, TaskType.BINARY]:
            class_counts = test_label.groupby(test_label).count()
            # Dict in format { Scorer : Dict { Class : Dict { Origin/Simple : score } } }
            results_dict = {}
            for scorer in scorers:
                model_dict = defaultdict(dict)
                for model_name, model_type, model_instance in models:
                    scorer_value = scorer(model_instance, test_dataset)
                    if isinstance(scorer_value, Number) or scorer_value is None:
                        model_dict[model_type] = scorer_value
                        if context.with_display:
                            display_array.append([model_name,
                                                  model_type,
                                                  scorer_value,
                                                  scorer.name,
                                                  test_label.count(),
                                                  ])
                    else:
                        for class_value, class_score in scorer_value.items():
                            # New labels which do not exists on the model gets nan as score, skips them.
                            # Also skips classes which are not in the test labels
                            if np.isnan(class_score) or class_value not in class_counts:
                                continue
                            model_dict[class_value][model_type] = class_score
                            if context.with_display:
                                classes_display_array.append([model_name,
                                                              model_type,
                                                              class_score,
                                                              scorer.name,
                                                              class_value,
                                                              class_counts[class_value]
                                                              ])
                results_dict[scorer.name] = model_dict
        else:
            # Dict in format { Scorer : Dict { Origin/Simple : score } }
            results_dict = {}
            for scorer in scorers:
                model_dict = defaultdict(dict)
                for model_name, model_type, model_instance in models:
                    score = scorer(model_instance, test_dataset)
                    model_dict[model_type] = score
                    if context.with_display:
                        display_array.append([model_name,
                                              model_type,
                                              score,
                                              scorer.name,
                                              test_label.count()
                                              ])
                results_dict[scorer.name] = model_dict

        figs = []
        if display_array:
            display_df = pd.DataFrame(
                display_array,
                columns=['Model', 'Type', 'Value', 'Metric', 'Number of samples']
            )

            # Plot the metrics in a graph, grouping by the model
            fig = (
                px.histogram(
                    display_df,
                    x='Model',
                    y='Value',
                    color='Model',
                    barmode='group',
                    facet_col='Metric',
                    facet_col_spacing=0.05,
                    hover_data=['Number of samples'])
                .update_xaxes(title=None)
                .update_yaxes(title=None, matches=None)
                .for_each_annotation(lambda a: a.update(text=a.text.split('=')[-1]))
                .for_each_yaxis(lambda yaxis: yaxis.update(showticklabels=True))
            )
            figs.append(fig)
        if classes_display_array:
            display_df = pd.DataFrame(
                classes_display_array,
                columns=['Model', 'Type', 'Value', 'Metric', 'Class', 'Number of samples']
            )

            # Plot the metrics in a graph, grouping by the model and class
            classes_fig = (
                px.histogram(
                    display_df,
                    x=['Class', 'Model'],
                    y='Value',
                    color='Model',
                    barmode='group',
                    facet_col='Metric',
                    facet_col_spacing=0.05,
                    hover_data=['Number of samples'])
                .update_xaxes(title=None, tickprefix='Class ', tickangle=60, type='category')
                .update_yaxes(title=None, matches=None)
                .for_each_annotation(lambda a: a.update(text=a.text.split('=')[-1]))
                .for_each_yaxis(lambda yaxis: yaxis.update(showticklabels=True))
            )
            figs.append(classes_fig)

        # For each scorer calculate perfect score in order to calculate later the ratio in conditions
        scorers_perfect = {scorer.name: scorer.score_perfect(test_dataset) for scorer in scorers}

        return CheckResult({'scores': results_dict,
                            'type': task_type,
                            'scorers_perfect': scorers_perfect,
                            }, display=figs)

    def config(self, include_version: bool = True, include_defaults: bool = True) -> 'CheckConfig':
        """Return check instance config."""
        if self.alternative_scorers is not None:
            for k, v in self.alternative_scorers.items():
                if callable(v):
                    reference = doclink(
                        'supported-metrics-by-string',
                        template='For a list of built-in scorers please refer to {link}. '
                    )
                    raise ValueError(
                        'Only built-in scorers are allowed when serializing check instances. '
                        f'{reference}Scorer name: {k}'
                    )
        return super().config(include_version, include_defaults=include_defaults)

    def _create_simple_model(self, train_ds: Dataset, task_type: TaskType):
        """Create a simple model of given type (random/constant/tree) to the given dataset.

        Parameters
        ----------
        train_ds : Dataset
            The training dataset object.
        task_type : TaskType
            the model type.

        Returns
        -------
        object
            Classifier Object

        Raises
        ------
        NotImplementedError
            If the strategy is not supported
        """
        if self.strategy == 'uniform':
            if task_type in [TaskType.BINARY, TaskType.MULTICLASS]:
                simple_model = ClassificationUniformModel()
            elif task_type == TaskType.REGRESSION:
                simple_model = RegressionUniformModel()
        elif self.strategy == 'stratified':
            simple_model = RandomModel()
        elif self.strategy == 'most_frequent':
            if task_type == TaskType.REGRESSION:
                simple_model = DummyRegressor(strategy='mean')
            elif task_type in {TaskType.BINARY, TaskType.MULTICLASS}:
                simple_model = DummyClassifier(strategy='most_frequent')
            else:
                raise DeepchecksValueError(f'Unknown task type - {task_type}')
        elif self.strategy == 'tree':
            if task_type == TaskType.REGRESSION:
                clf = DecisionTreeRegressor(
                    max_depth=self.max_depth,
                    random_state=self.random_state
                )
            elif task_type in {TaskType.BINARY, TaskType.MULTICLASS}:
                clf = DecisionTreeClassifier(
                    max_depth=self.max_depth,
                    random_state=self.random_state,
                    class_weight='balanced'
                )
            else:
                raise DeepchecksValueError(f'Unknown task type - {task_type}')

            simple_model = Pipeline([('scaler', ScaledNumerics(train_ds.cat_features, max_num_categories=10)),
                                     ('tree-model', clf)])
        else:
            raise DeepchecksValueError(
                f'Unknown model type - {self.strategy}, expected to be one of '
                f"['uniform', 'stratified', 'most_frequent', 'tree'] "
                f"but instead got {self.strategy}"  # pylint: disable=inconsistent-quotes
            )

        simple_model.fit(train_ds.data[train_ds.features], train_ds.data[train_ds.label_name])
        return simple_model

    def add_condition_gain_greater_than(self,
                                        min_allowed_gain: float = 0.1,
                                        classes: List[Hashable] = None,
                                        average: bool = False):
        """Add condition - require minimum allowed gain between the model and the simple model.

        Parameters
        ----------
        min_allowed_gain : float , default: 0.1
            Minimum allowed gain between the model and the simple model -
            gain is: difference in performance / (perfect score - simple score)
        classes : List[Hashable] , default: None
            Used in classification models to limit condition only to given classes.
        average : bool , default: False
            Used in classification models to flag if to run condition on average of classes, or on
            each class individually
        """
        name = f'Model performance gain over simple model is greater than {format_percent(min_allowed_gain)}'
        if classes:
            name = name + f' for classes {str(classes)}'
        return self.add_condition(name,
                                  condition,
                                  include_classes=classes,
                                  min_allowed_gain=min_allowed_gain,
                                  max_gain=self.max_gain,
                                  average=average)


def condition(result: Dict, include_classes=None, average=False, max_gain=None, min_allowed_gain=None) -> \
        ConditionResult:
    scores = result['scores']
    task_type = result['type']
    scorers_perfect = result['scorers_perfect']

    passed_condition = True
    if task_type in [TaskType.MULTICLASS, TaskType.BINARY] and not average:
        passed_metrics = {}
        failed_classes = defaultdict(dict)
        perfect_metrics = []
        for metric, classes_scores in scores.items():
            gains = {}
            metric_passed = True
            for clas, models_scores in classes_scores.items():
                # Skip if class is not in class list
                if include_classes is not None and clas not in include_classes:
                    continue

                # If origin model is perfect, skip the gain calculation
                if models_scores['Origin'] == scorers_perfect[metric]:
                    continue

                gains[clas] = get_gain(models_scores['Simple'],
                                       models_scores['Origin'],
                                       scorers_perfect[metric],
                                       max_gain)
                # Save dict of failed classes and metrics gain
                if gains[clas] <= min_allowed_gain:
                    failed_classes[clas][metric] = format_percent(gains[clas])
                    metric_passed = False

            if metric_passed and gains:
                avg_gain = sum(gains.values()) / len(gains)
                passed_metrics[metric] = format_percent(avg_gain)
            elif metric_passed and not gains:
                perfect_metrics.append(metric)

        if failed_classes:
            msg = f'Found classes with failed metric\'s gain: {dict(failed_classes)}'
            passed_condition = False
        elif passed_metrics:
            msg = f'All classes passed, average gain for metrics: {passed_metrics}'
        else:
            msg = f'Found metrics with perfect score, no gain is calculated: {perfect_metrics}'
    else:
        passed_metrics = {}
        failed_metrics = {}
        perfect_metrics = []
        if task_type in [TaskType.MULTICLASS, TaskType.BINARY]:
            scores = average_scores(scores, include_classes)
        for metric, models_scores in scores.items():
            # If origin model is perfect, skip the gain calculation
            if models_scores['Origin'] == scorers_perfect[metric]:
                perfect_metrics.append(metric)
                continue
            gain = get_gain(models_scores['Simple'],
                            models_scores['Origin'],
                            scorers_perfect[metric],
                            max_gain)
            if gain <= min_allowed_gain:
                failed_metrics[metric] = format_percent(gain)
            else:
                passed_metrics[metric] = format_percent(gain)
        if failed_metrics:
            msg = f'Found failed metrics: {failed_metrics}'
            passed_condition = False
        elif passed_metrics:
            msg = f'All metrics passed, metric\'s gain: {passed_metrics}'
        else:
            msg = f'Found metrics with perfect score, no gain is calculated: {perfect_metrics}'

    category = ConditionCategory.PASS if passed_condition else ConditionCategory.FAIL
    return ConditionResult(category, msg)


def average_scores(scores, include_classes):
    result = {}
    for metric, classes_scores in scores.items():
        origin_score = 0
        simple_score = 0
        total = 0
        for clas, models_scores in classes_scores.items():
            # Skip if class is not in class list
            if include_classes is not None and clas not in include_classes:
                continue
            origin_score += models_scores['Origin']
            simple_score += models_scores['Simple']
            total += 1

        result[metric] = {
            'Origin': origin_score / total,
            'Simple': simple_score / total
        }
    return result
