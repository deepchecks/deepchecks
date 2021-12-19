# ----------------------------------------------------------------------------
# Copyright (C) 2021 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""Boosting overfit check module."""
from copy import deepcopy
from typing import Callable, Union

from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from deepchecks import Dataset, CheckResult, TrainTestBaseCheck, ConditionResult
from deepchecks.utils.metrics import task_type_check, DEFAULT_METRICS_DICT, validate_scorer, DEFAULT_SINGLE_METRIC
from deepchecks.utils.strings import format_percent
from deepchecks.utils.validation import validate_model
from deepchecks.utils.model import get_model_of_pipeline
from deepchecks.utils.plot import colors
from deepchecks.errors import DeepchecksValueError


__all__ = ['BoostingOverfit']


class PartialBoostingModel:
    """Wrapper for boosting models which limits the number of estimators being used in the prediction."""

    def __init__(self, model, step):
        """Construct wrapper for model with `predict` and `predict_proba` methods.

        Args:
            model: boosting model to wrap.
            step: Number of iterations/estimators to limit the model on predictions.
        """
        self.model_class = get_model_of_pipeline(model).__class__.__name__
        self.step = step
        if self.model_class in ['AdaBoostClassifier', 'GradientBoostingClassifier', 'AdaBoostRegressor',
                                'GradientBoostingRegressor']:
            self.model = deepcopy(model)
            if isinstance(model, Pipeline):
                internal_estimator = get_model_of_pipeline(self.model)
                internal_estimator.estimators_ = internal_estimator.estimators_[:self.step]
            else:
                self.model.estimators_ = self.model.estimators_[:self.step]
        else:
            self.model = model

    def predict_proba(self, x):
        if self.model_class in ['AdaBoostClassifier', 'GradientBoostingClassifier']:
            return self.model.predict_proba(x)
        elif self.model_class == 'LGBMClassifier':
            return self.model.predict_proba(x, num_iteration=self.step)
        elif self.model_class == 'XGBClassifier':
            return self.model.predict_proba(x, iteration_range=(0, self.step))
        elif self.model_class == 'CatBoostClassifier':
            return self.model.predict_proba(x, ntree_end=self.step)
        else:
            raise DeepchecksValueError(f'Unsupported model of type: {self.model_class}')

    def predict(self, x):
        if self.model_class in ['AdaBoostClassifier', 'GradientBoostingClassifier', 'AdaBoostRegressor',
                                'GradientBoostingRegressor']:
            return self.model.predict(x)
        elif self.model_class in ['LGBMClassifier', 'LGBMRegressor']:
            return self.model.predict(x, num_iteration=self.step)
        elif self.model_class in ['XGBClassifier', 'XGBRegressor']:
            return self.model.predict(x, iteration_range=(0, self.step))
        elif self.model_class in ['CatBoostClassifier', 'CatBoostRegressor']:
            return self.model.predict(x, ntree_end=self.step)
        else:
            raise DeepchecksValueError(f'Unsupported model of type: {self.model_class}')

    @classmethod
    def n_estimators(cls, model):
        model = get_model_of_pipeline(model)
        model_class = model.__class__.__name__
        if model_class in ['AdaBoostClassifier', 'GradientBoostingClassifier', 'AdaBoostRegressor',
                           'GradientBoostingRegressor']:
            return len(model.estimators_)
        elif model_class in ['LGBMClassifier', 'LGBMRegressor']:
            return model.n_estimators
        elif model_class in ['XGBClassifier', 'XGBRegressor']:
            return model.n_estimators
        elif model_class in ['CatBoostClassifier', 'CatBoostRegressor']:
            return model.tree_count_
        else:
            raise DeepchecksValueError(f'Unsupported model of type: {model_class}')


def partial_score(scorer, dataset, model, step):
    partial_model = PartialBoostingModel(model, step)
    return scorer(partial_model, dataset.features_columns, dataset.label_col)


def calculate_steps(num_steps, num_estimators):
    """Calculate steps (integers between 1 to num_estimators) to work on."""
    if num_steps >= num_estimators:
        return list(range(1, num_estimators + 1))
    if num_steps <= 5:
        steps_percents = np.linspace(0, 1.0, num_steps + 1)[1:]
        steps_numbers = np.ceil(steps_percents * num_estimators)
        steps_set = {int(s) for s in steps_numbers}
    else:
        steps_percents = np.linspace(5 / num_estimators, 1.0, num_steps - 4)[1:]
        steps_numbers = np.ceil(steps_percents * num_estimators)
        steps_set = {int(s) for s in steps_numbers}
        # We want to forcefully take the first 5 estimators, since they have the largest affect on the model performance
        steps_set.update({1, 2, 3, 4, 5})

    return sorted(steps_set)


class BoostingOverfit(TrainTestBaseCheck):
    """Check for overfit occurring when increasing the number of iterations in boosting models.

    The check runs a pred-defined number of steps, and in each step it limits the boosting model to use up to X
    estimators (number of estimators is monotonic increasing). It plots the given metric calculated for each step for
    both the train dataset and the test dataset.

    Args:
        metric (Union[Callable, str]): Metric to use verify the model, either function or sklearn scorer name.
        metric_name (str): Name to be displayed in the plot on y-axis. must be used together with 'metric'
        num_steps (int): Number of splits of the model iterations to check.
    """

    def __init__(self, metric: Union[Callable, str] = None, metric_name: str = None, num_steps: int = 20):
        super().__init__()
        self.metric = metric
        self.metric_name = metric_name
        self.num_steps = num_steps

    def run(self, train_dataset, test_dataset, model=None) -> CheckResult:
        """Run check.

        Args:
            train_dataset (Dataset):
            test_dataset (Dataset):
            model: Boosting model.

        Returns:
            The metric value on the test dataset.
        """
        return self._boosting_overfit(train_dataset, test_dataset, model=model)

    def _boosting_overfit(self, train_dataset: Dataset, test_dataset: Dataset, model) -> CheckResult:
        # Validate params
        if self.metric_name is not None and self.metric is None:
            raise DeepchecksValueError('Can not have metric_name without metric')
        if not isinstance(self.num_steps, int) or self.num_steps < 2:
            raise DeepchecksValueError('num_steps must be an integer larger than 1')
        Dataset.validate_dataset(train_dataset)
        Dataset.validate_dataset(test_dataset)
        train_dataset.validate_label()
        test_dataset.validate_label()
        train_dataset.validate_shared_features(test_dataset)
        train_dataset.validate_shared_label(test_dataset)
        validate_model(train_dataset, model)

        # Get default metric
        model_type = task_type_check(model, train_dataset)
        if self.metric is not None:
            scorer = validate_scorer(self.metric, model, train_dataset)
            metric_name = self.metric_name or self.metric if isinstance(self.metric, str) else 'User metric'
        else:
            metric_name = DEFAULT_SINGLE_METRIC[model_type]
            scorer = DEFAULT_METRICS_DICT[model_type][metric_name]

        # Get number of estimators on model
        num_estimators = PartialBoostingModel.n_estimators(model)
        estimator_steps = calculate_steps(self.num_steps, num_estimators)

        train_scores = []
        test_scores = []
        for step in estimator_steps:
            train_scores.append(partial_score(scorer, train_dataset, model, step))
            test_scores.append(partial_score(scorer, test_dataset, model, step))

        def display_func():
            _, axes = plt.subplots(1, 1, figsize=(7, 4))
            axes.set_xlabel('Number of boosting iterations')
            axes.set_ylabel(metric_name)
            axes.grid()
            axes.plot(estimator_steps, np.array(train_scores), 'o-', color=colors['Train'], label='Training score')
            axes.plot(estimator_steps, np.array(test_scores), 'o-', color=colors['Test'], label='Test score')
            axes.legend(loc='best')
            # Display x ticks as integers
            axes.xaxis.set_major_locator(MaxNLocator(integer=True))

        result = {'test': test_scores, 'train': train_scores}
        return CheckResult(result, display=display_func, header='Boosting Overfit')

    def add_condition_test_score_percent_decline_not_greater_than(self, threshold: float = 0.05):
        """Add condition.

        Percent of decline between the maximal score achieved in any boosting iteration and the score achieved in the
        last iteration ("regular" model score) is not above given threshold.

        Args:
            threshold (float): Maximum percentage decline allowed (value 0 and above)
        """
        def condition(result: dict):
            max_score = max(result['test'])
            last_score = result['test'][-1]
            pct_diff = (max_score - last_score) / abs(max_score)

            if pct_diff > threshold:
                message = f'Found metric decline of: -{format_percent(pct_diff)}'
                return ConditionResult(False, message)
            else:
                return ConditionResult(True)

        return self.add_condition(f'Test score decline is not greater than {format_percent(threshold)}', condition)
