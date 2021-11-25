"""Module containing naive comparison check."""
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
from deepchecks.string_utils import format_number

from deepchecks import CheckResult, Dataset
from deepchecks.base.check import ConditionResult, TrainTestBaseCheck
from deepchecks.metric_utils import DEFAULT_METRICS_DICT, DEFAULT_SINGLE_METRIC, task_type_check, ModelType, \
    validate_scorer, get_metrics_ratio
from deepchecks.utils import model_type_validation

__all__ = ['NaiveModelComparison']


class DummyModel:
    @staticmethod
    def predict(a):
        return a

    @staticmethod
    def predict_proba(a):
        return a


def find_score(train_ds: Dataset, test_ds: Dataset, task_type: ModelType, model,
              naive_model_type: str, metric = None, metric_name = None):
    """Find the naive model score for given metric.

    Args:
        train_ds (Dataset): The training dataset object. Must contain an index.
        test_ds (Dataset): The test dataset object. Must contain an index.
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
    test_df = test_ds.data

    np.random.seed(0)

    if naive_model_type == 'random':
        naive_pred = np.random.choice(train_ds.label_col(), test_df.shape[0])

    elif naive_model_type == 'statistical':
        if task_type == ModelType.REGRESSION:
            naive_pred = np.array([np.mean(train_ds.label_col())] * len(test_df))

        elif task_type in (ModelType.BINARY, ModelType.MULTICLASS):
            counts = train_ds.label_col().mode()
            naive_pred = np.array([counts.index[0]] * len(test_df))

    else:
        raise NotImplementedError(f"expected to be one of ['random', 'statistical'] \
                                   but instead got {naive_model_type}")

    y_test = test_ds.label_col()

    if metric is not None:
        scorer = validate_scorer(metric, model, train_ds)
        metric_name = metric_name or metric if isinstance(metric, str) else 'User metric'
    else:
        metric_name = DEFAULT_SINGLE_METRIC[task_type]
        scorer = DEFAULT_METRICS_DICT[task_type][metric_name]

    naive_metric = scorer(DummyModel, naive_pred, y_test)
    pred_metric = scorer(model, test_ds.features_columns(), y_test)

    return naive_metric, pred_metric, metric_name


class NaiveModelComparison(TrainTestBaseCheck):
    """Compare naive model score to given model score.

    Args:
        naive_model_type (str = 'random'):  Type of the naive model ['random', 'statistical'].
        metric: a custom metric given by user.
        metric_name: name of a default metric.
        maximum_ratio: the ratio can be up to infinity so choose maximum value to limit to.
    """

    def __init__(self, naive_model_type: str = 'statistical', metric=None, metric_name=None, maximum_ratio: int = 10):
        super().__init__()
        self.naive_model_type = naive_model_type
        self.metric = metric
        self.metric_name = metric_name
        self.maximum_ratio = maximum_ratio

    def run(self, train_dataset, test_dataset, model) -> CheckResult:
        """Run check.

        Args:
            train_dataset (Dataset): The training dataset object. Must contain a label.
            test_dataset (Dataset): The test dataset object. Must contain a label.
            model (BaseEstimator): A scikit-learn-compatible fitted estimator instance.

        Returns:
            CheckResult: value is a Dict of: given_model_score, naive_model_score, ratio
                         ratio is given model / naive model (if the metric returns negative values we devied 1 by it)
                         if ratio is infinite 99999 is returned

        Raises:
            DeepchecksValueError: If the object is not a Dataset instance.
        """
        return self._naive_model_comparison(train_dataset, test_dataset, model)

    def _naive_model_comparison(self, train_dataset: Dataset, test_dataset: Dataset, model):
        func_name = self.__class__.__name__
        Dataset.validate_dataset(train_dataset, func_name)
        Dataset.validate_dataset(test_dataset, func_name)
        train_dataset.validate_label(func_name)
        test_dataset.validate_label(func_name)
        model_type_validation(model)

        naive_metric, pred_metric, metric_name = find_score(train_dataset, test_dataset,
                                                            task_type_check(model, train_dataset), model,
                                                            self.naive_model_type, self.metric,
                                                            self.metric_name)

        ratio = get_metrics_ratio(naive_metric, pred_metric, self.maximum_ratio)

        text = f'The given model performs {format_number(ratio)} times compared to' \
               f' the naive model using the {metric_name} metric.<br>' \
               f'{type(model).__name__} model prediction has achieved a score of {format_number(pred_metric)} ' \
               f'compared to Naive {self.naive_model_type} prediction ' \
               f'which achieved a score of {format_number(naive_metric)} on tested data.'

        def display_func():
            fig = plt.figure()
            ax = fig.add_axes([0, 0, 1, 1])
            models = [f'Naive model - {self.naive_model_type}', f'{type(model).__name__} model']
            metrics_results = [naive_metric, pred_metric]
            ax.bar(models, metrics_results)
            ax.set_ylabel(metric_name)

        return CheckResult({'given_model_score': pred_metric, 'naive_model_score': naive_metric,
                            'ratio': ratio},
                           check=self.__class__, display=[text, display_func])

    def add_condition_ratio_not_less_than(self, min_allowed_ratio: float = 1.1):
        """Add condition - require min allowed ratio between the naive and the given model.

        Args:
            min_allowed_ratio (float): Min allowed ratio between the naive and the given model -
            ratio is given model / naive model (if the metric returns negative values we devied 1 by it)
        """
        def condition(result: Dict) -> ConditionResult:
            ratio = result['ratio']
            if ratio < min_allowed_ratio:
                return ConditionResult(False,
                                       f'The given model performs {format_number(ratio)} times compared'
                                       f'to the naive model using the given metric')
            else:
                return ConditionResult(True)

        return self.add_condition(f'Ratio not less than {format_number(min_allowed_ratio)} '
                                  f'between the given model\'s result and the naive model\'s result',
                                  condition)
