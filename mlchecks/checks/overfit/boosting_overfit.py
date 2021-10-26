"""Boosting overfit check module."""
from copy import deepcopy
from typing import Callable, Union
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import numpy as np

from mlchecks import Dataset, CheckResult, TrainValidationBaseCheck
from mlchecks.metric_utils import task_type_check, DEFAULT_METRICS_DICT, validate_scorer, DEFAULT_SINGLE_METRIC
from mlchecks.utils import MLChecksValueError

__all__ = ['boosting_overfit', 'BoostingOverfit']


class PartialBoostingModel:
    """Wrapper for boosting models which limits the number of estimators being used in the prediction."""

    def __init__(self, model, step):
        """Construct wrapper for model with `predict` and `predict_proba` methods.

        Args:
            model: boosting model to wrap.
            step: Number of iterations/estimators to limit the model on predictions.
        """
        self.model_class = model.__class__.__name__
        self.step = step
        if self.model_class in ['AdaBoostClassifier', 'GradientBoostingClassifier', 'AdaBoostRegressor',
                                'GradientBoostingRegressor']:
            self.model = deepcopy(model)
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
            raise MLChecksValueError(f'Unsupported model of type: {self.model_class}')

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
            raise MLChecksValueError(f'Unsupported model of type: {self.model_class}')

    @classmethod
    def n_estimators(cls, model):
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
            raise MLChecksValueError(f'Unsupported model of type: {model_class}')


def partial_score(scorer, dataset, model, step):
    partial_model = PartialBoostingModel(model, step)
    return scorer(partial_model, dataset.features_columns(), dataset.label_col())


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


def boosting_overfit(train_dataset: Dataset, validation_dataset: Dataset, model, metric: Union[Callable, str] = None,
                     metric_name: str = None, num_steps: int = 20) \
        -> CheckResult:
    f"""Check for overfit occurring by number of iterations in boosting models.
    
    The check runs a defined number of steps, and in each step is limiting the boosting model to use up to X estimators
    (number of estimators is monotonic increasing). It plots the given metric in each step for both the train dataset
    and the validation dataset.

    Args:
        train_dataset (Dataset):
        validation_dataset (Dataset):
        model: Boosting model.
        metric (Union[Callable, str]): Metric to use verify the model, either function or sklearn scorer name.
        metric_name (str): Name to be displayed in the plot on y-axis. must be used together with 'metric'
        num_steps (int): Number of splits of the model iterations to check.

    Returns:
        The metric value on the validation dataset.
    """
    # Validate params
    self = boosting_overfit
    if metric_name is not None and metric is None:
        raise MLChecksValueError('Can not have metric_name without metric')
    if not isinstance(num_steps, int) or num_steps < 2:
        raise MLChecksValueError('num_steps must be an integer larger than 1')
    Dataset.validate_dataset(train_dataset, self.__name__)
    Dataset.validate_dataset(validation_dataset, self.__name__)
    train_dataset.validate_label(self.__name__)
    validation_dataset.validate_label(self.__name__)
    train_dataset.validate_shared_features(validation_dataset, self.__name__)
    train_dataset.validate_shared_label(validation_dataset, self.__name__)
    train_dataset.validate_model(model)

    # Get default metric
    model_type = task_type_check(model, train_dataset)
    if metric is not None:
        scorer = validate_scorer(metric, model, train_dataset)
        metric_name = metric_name or metric if isinstance(metric, str) else 'User metric'
    else:
        metric_name = DEFAULT_SINGLE_METRIC[model_type]
        scorer = DEFAULT_METRICS_DICT[model_type][metric_name]

    # Get number of estimators on model
    num_estimators = PartialBoostingModel.n_estimators(model)
    estimator_steps = calculate_steps(num_steps, num_estimators)

    train_scores = []
    val_scores = []
    for step in estimator_steps:
        train_scores.append(partial_score(scorer, train_dataset, model, step))
        val_scores.append(partial_score(scorer, validation_dataset, model, step))

    def display_func():
        _, axes = plt.subplots(1, 1, figsize=(7, 4))
        axes.set_xlabel('Number of boosting iterations')
        axes.set_ylabel(metric_name)
        axes.grid()
        axes.plot(estimator_steps, np.array(train_scores), 'o-', color='r', label='Training score')
        axes.plot(estimator_steps, np.array(val_scores), 'o-', color='g', label='Validation score')
        axes.legend(loc='best')
        # Display x ticks as integers
        axes.xaxis.set_major_locator(MaxNLocator(integer=True))

    return CheckResult(val_scores[-1], check=boosting_overfit, display=display_func)


class BoostingOverfit(TrainValidationBaseCheck):
    f"""Check for overfit occurring by number of iterations in boosting models.

    The check runs a defined number of steps, and in each step is limiting the boosting model to use up to X estimators
    (number of estimators is monotonic increasing). It plots the given metric in each step for both the train dataset
    and the validation dataset.
    """

    def __init__(self, **kwargs):
        """Construct instance with given parameters

        Args:
            metric (Union[Callable, str]): Metric to use verify the model, either function or sklearn scorer name.
            metric_name (str): Name to be displayed in the plot on y-axis. must be used together with 'metric'
            num_steps (int): Number of splits of the model iterations to check.
        """
        super().__init__(**kwargs)

    def run(self, train_dataset, validation_dataset, model=None) -> CheckResult:
        """Run boosting_overfit on given parameters.

        Args:
            train_dataset (Dataset):
            validation_dataset (Dataset):
            model: Boosting model.
        """
        return boosting_overfit(train_dataset, validation_dataset, model=model, **self.params)
