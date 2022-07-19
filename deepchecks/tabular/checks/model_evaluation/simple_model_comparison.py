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
from typing import Callable, Dict, Hashable, List

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
from deepchecks.utils.metrics import get_gain
from deepchecks.utils.simple_models import ClassificationUniformModel, RandomModel, RegressionUniformModel
from deepchecks.utils.strings import format_percent

__all__ = ['SimpleModelComparison']

_allowed_strategies = (
    'stratified',
    'most_frequent',
    'uniform',
    'tree'
)

_depr_strategies = {
    'random': 'stratified',
    'constant': 'most_frequent',
}


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
    simple_model_type : str , default: most_frequent
        Deprecated. Please use strategy instead.
    alternative_scorers : Dict[str, Callable], default: None
        An optional dictionary of scorer title to scorer functions/names. If none given, using default scorers.
        For description about scorers see Notes below.
    max_gain : float , default: 50
        the maximum value for the gain value, limits from both sides [-max_gain, max_gain]
    max_depth : int , default: 3
        the max depth of the tree (used only if simple model type is tree).
    random_state : int , default: 42
        the random state (used only if simple model type is tree or random).

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

    def __init__(
        self,
        strategy: str = 'most_frequent',
        simple_model_type: str = None,
        alternative_scorers: Dict[str, Callable] = None,
        max_gain: float = 50,
        max_depth: int = 3,
        random_state: int = 42,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.user_scorers = alternative_scorers
        self.max_gain = max_gain
        self.max_depth = max_depth
        self.random_state = random_state

        if simple_model_type is not None:
            warnings.warn(
                f'{self.__class__.__name__}: simple_model_type is deprecated. please use strategy instead.',
                DeprecationWarning
            )
            self.strategy = simple_model_type
        else:
            self.strategy = strategy

        if self.strategy in _depr_strategies:
            warnings.warn(
                f'{self.__class__.__name__}: strategy {self.strategy} is deprecated. '
                f'please use { _depr_strategies[self.strategy] } instead.',
                DeprecationWarning
            )
            self.strategy = _depr_strategies[self.strategy]

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
        train_dataset = context.train
        test_dataset = context.test
        test_label = test_dataset.label_col
        task_type = context.task_type
        model = context.model

        # If user defined scorers used them, else use a single scorer
        if self.user_scorers:
            scorers = context.get_scorers(self.user_scorers, use_avg_defaults=False)
        else:
            scorers = [context.get_single_scorer(use_avg_defaults=False)]

        simple_model = self._create_simple_model(train_dataset, task_type)

        models = [
            (f'{type(model).__name__} model', 'Origin', model),
            (f'Simple model - {self.strategy}', 'Simple', simple_model)
        ]

        # Multiclass have different return type from the scorer, list of score per class instead of single score
        if task_type in [TaskType.MULTICLASS, TaskType.BINARY]:
            n_samples = test_label.groupby(test_label).count()
            classes = [clazz for clazz in test_dataset.classes if clazz in train_dataset.classes]

            display_array = []
            # Dict in format { Scorer : Dict { Class : Dict { Origin/Simple : score } } }
            results_dict = {}
            for scorer in scorers:
                model_dict = defaultdict(dict)
                for model_name, model_type, model_instance in models:
                    for class_score, class_value in zip(scorer(model_instance, test_dataset), classes):
                        model_dict[class_value][model_type] = class_score
                        if context.with_display:
                            display_array.append([model_name,
                                                  model_type,
                                                  class_score,
                                                  scorer.name,
                                                  class_value,
                                                  n_samples[class_value]
                                                  ])
                results_dict[scorer.name] = model_dict

            if display_array:
                display_df = pd.DataFrame(
                    display_array,
                    columns=['Model', 'Type', 'Value', 'Metric', 'Class', 'Number of samples']
                )

                # Plot the metrics in a graph, grouping by the model and class
                fig = (
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
            else:
                fig = None

        else:
            classes = None

            display_array = []
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
            else:
                fig = None

        # For each scorer calculate perfect score in order to calculate later the ratio in conditions
        scorers_perfect = {scorer.name: scorer.score_perfect(test_dataset) for scorer in scorers}

        return CheckResult({'scores': results_dict,
                            'type': task_type,
                            'scorers_perfect': scorers_perfect,
                            'classes': classes
                            }, display=fig)

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
        np.random.seed(self.random_state)

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
