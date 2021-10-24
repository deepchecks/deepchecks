from copy import deepcopy
from typing import Callable
import matplotlib.pyplot as plt
import numpy as np

from mlchecks import Dataset, CheckResult
from mlchecks.metric_utils import ModelType, task_type_check, DEFAULT_METRICS_DICT, validate_scorer
from mlchecks.utils import MLChecksValueError

__all__ = ['boosting_overfit']

DEFAULT_SINGLE_METRIC = {
    ModelType.BINARY: 'Accuracy',
    ModelType.MULTICLASS: 'Accuracy',
    ModelType.REGRESSION: 'RMSE'
}


class PartialBoostingModel:
    """    """

    def __init__(self, model, step):
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
            raise Exception()

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
            raise Exception()

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
            raise Exception()


def make_score(scorer, dataset, model, step):
    partial_model = PartialBoostingModel(model, step)
    return scorer(partial_model, dataset.features_columns(), dataset.label_col())


def boosting_overfit(train_dataset: Dataset, validation_dataset: Dataset, model, metric: Callable = None,
                     metric_name: str = None, num_steps: int = 20) \
        -> CheckResult:
    """Test for overfit in boosting models."""
    # Validate params
    if (metric is None) ^ (metric_name is None):
        raise MLChecksValueError('Must have both metric and metric_name defined or None together')
    if not isinstance(num_steps, int) or num_steps < 2:
        raise MLChecksValueError('num_steps must be an integer larger than 1')

    # Get default metric
    model_type = task_type_check(model, train_dataset)
    metric_name = metric_name or DEFAULT_SINGLE_METRIC[model_type]
    if metric is not None:
        scorer = validate_scorer(metric, model, train_dataset)
    else:
        scorer = DEFAULT_METRICS_DICT[model_type][metric_name]

    # Get number of estimators on model
    num_estimators = PartialBoostingModel.n_estimators(model)
    # Calculate estimator steps
    steps_percents = np.linspace(0, 1.0, num_steps)[1:]
    steps_numbers = np.ceil(steps_percents * num_estimators)
    estimator_steps = sorted({int(s) for s in steps_numbers})

    train_scores = []
    val_scores = []
    for step in estimator_steps:
        train_scores.append(make_score(scorer, train_dataset, model, step))
        val_scores.append(make_score(scorer, validation_dataset, model, step))

    def display_func():
        estimator_percents = [x / num_estimators for x in estimator_steps]
        fig, axes = plt.subplots(1, 1, figsize=(7, 4))
        axes.set_xlabel('Percent estimators used')
        axes.set_ylabel(metric_name)
        axes.grid()
        axes.plot(estimator_percents, np.array(train_scores), 'o-', color="r",
                  label="Training score")
        axes.plot(estimator_percents, np.array(val_scores), 'o-', color="g",
                  label="Validation score")
        axes.legend(loc="best")

    return CheckResult(val_scores[-1], check=boosting_overfit, display=display_func)
