"""Module containing naive comparison check."""
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
from deepchecks.string_utils import format_columns_for_condition, format_number

from deepchecks import CheckResult, Dataset
from deepchecks.base.check import ConditionResult, TrainValidationBaseCheck
from deepchecks.metric_utils import DEFAULT_METRICS_DICT, DEFAULT_SINGLE_METRIC, task_type_check, ModelType, validate_scorer
from deepchecks.utils import model_type_validation

__all__ = ['NaiveModelComparison']


class DummyModel:
    @staticmethod
    def predict(a):
        return a

    @staticmethod
    def predict_proba(a):
        return a


def find_score(train_ds: Dataset, val_ds: Dataset, task_type: ModelType, model,
              naive_model_type: str, metric = None, metric_name = None):
    """Find the naive model score for given metric.

    Args:
        train_ds (Dataset): The training dataset object. Must contain an index.
        val_ds (Dataset): The validation dataset object. Must contain an index.
        task_type (ModelType): the model type.
        model (BaseEstimator): A scikit-learn-compatible fitted estimator instance.
        naive_model_type (str): Type of the naive model ['random', 'statistical'].
                                random is random label from train,
                                statistical is mean for regression or
                                the label that appears most often for classification
        metric: a custom metric given by user.
        metric_name: name of a default metric.
    Returns:
        float: p value for the key.

    Raises:
        NotImplementedError: If the naive_model_type is not supported

    """
    val_df = val_ds.data

    np.random.seed(0)

    if naive_model_type == 'random':
        naive_pred = np.random.choice(train_ds.label_col(), val_df.shape[0])

    elif naive_model_type == 'statistical':
        if task_type == ModelType.REGRESSION:
            naive_pred = np.array([np.mean(train_ds.label_col())] * len(val_df))

        elif task_type in (ModelType.BINARY, ModelType.MULTICLASS):
            counts = train_ds.label_col().mode()
            naive_pred = np.array([counts.index[0]] * len(val_df))

    else:
        raise NotImplementedError(f"expected to be one of ['random', 'statistical'] \
                                   but instead got {naive_model_type}")

    y_val = val_ds.label_col()

    if metric is not None:
        scorer = validate_scorer(metric, model, train_ds)
        metric_name = metric_name or metric if isinstance(metric, str) else 'User metric'
    else:
        metric_name = DEFAULT_SINGLE_METRIC[task_type]
        scorer = DEFAULT_METRICS_DICT[task_type][metric_name]

    naive_metric = scorer(DummyModel, naive_pred, y_val)
    pred_metric = scorer(model, val_ds.features_columns(), y_val)

    return naive_metric, pred_metric, metric_name


class NaiveModelComparison(TrainValidationBaseCheck):
    """Compare naive model score to given model score."""

    def __init__(self, naive_model_type: str = 'statistical', metric=None, metric_name=None):
        """Initialize the NaiveModelComparison check.

        Args:
            naive_model_type (str = 'random'):  Type of the naive model ['random', 'statistical'].
            metric: a custom metric given by user.
            metric_name: name of a default metric.
        """
        super().__init__()
        self.naive_model_type = naive_model_type
        self.metric = metric
        self.metric_name = metric_name

    def run(self, train_dataset, validation_dataset, model) -> CheckResult:
        """Run check.

        Args:
            train_dataset (Dataset): The training dataset object. Must contain a label.
            validation_dataset (Dataset): The validation dataset object. Must contain a label.
            model (BaseEstimator): A scikit-learn-compatible fitted estimator instance.

        Returns:
            CheckResult: value is ratio between model prediction to naive prediction

        Raises:
            MLChecksValueError: If the object is not a Dataset instance.
        """
        return self._naive_model_comparison(train_dataset, validation_dataset, model)

    def _naive_model_comparison(self, train_dataset: Dataset, validation_dataset: Dataset, model):
        func_name = self.__class__.__name__
        Dataset.validate_dataset(train_dataset, func_name)
        Dataset.validate_dataset(validation_dataset, func_name)
        train_dataset.validate_label(func_name)
        validation_dataset.validate_label(func_name)
        model_type_validation(model)

        naive_metric, pred_metric, metric_name = find_score(train_dataset, validation_dataset,
                                                            task_type_check(model, train_dataset), model,
                                                            self.naive_model_type, self.metric,
                                                            self.metric_name)

        effective_ratio = ratio = naive_metric / pred_metric
        if naive_metric < 0 and pred_metric < 0:
            effective_ratio = 1 / ratio

        text = f'The naive model is {format_number(effective_ratio)} times as effective as the ' \
               f'checked model using the {metric_name} metric.<br>' \
               f'{type(model).__name__} model prediction has achieved {format_number(pred_metric)} ' \
               f'compared to Naive {self.naive_model_type} prediction ' \
               f'which achieved {format_number(naive_metric)} on tested data.'

        def display_func():
            fig = plt.figure()
            ax = fig.add_axes([0, 0, 1, 1])
            models = [f'Naive model - {self.naive_model_type}', f'{type(model).__name__} model']
            metrics_results = [naive_metric, pred_metric]
            ax.bar(models, metrics_results)
            ax.set_ylabel(metric_name)

        return CheckResult({'given_model_score': pred_metric, 'naive_model_score': naive_metric,
                            'effective_ratio': effective_ratio},
                           check=self.__class__, display=[text, display_func])


    def add_condition_max_effective_ratio(self, max_allowed_effective_ratio: float = 0.7):
        """Add condition - require column not to have more than given number of different null values.

        Args:
            max_allowed_null_types (int): Number of different null value types which is the maximum allowed.
        """
        def condition(result: Dict) -> ConditionResult:
            effective_ratio = result['effective_ratio']
            if effective_ratio > max_allowed_effective_ratio:
                return ConditionResult(False,
                                        f'The naive model is {format_number(effective_ratio)} times as effective as the ' \
                                        f'checked model using the given metric, which is more than the allowed ration of: '
                                        f'{format_number(max_allowed_effective_ratio)}')
            else:
                return ConditionResult(True)

        return self.add_condition(f'Not more than {format_number(max_allowed_effective_ratio)} effective ratio '
                                  f'between the naive model\'s result and the checked model\'s result',
                                  condition)

