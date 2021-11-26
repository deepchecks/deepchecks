"""The class performance imbalance check."""
import typing as t
from functools import partial
from collections import defaultdict

import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score
from sklearn.utils.multiclass import unique_labels

from deepchecks import SingleDatasetBaseCheck, Dataset, CheckResult, ConditionResult
from deepchecks.metric_utils import task_type_validation, ModelType
from deepchecks.string_utils import format_percent
from deepchecks.utils import DeepchecksValueError


__all__ = ['ClassPerformanceImbalanceCheck']


MetricFunc = t.Callable[
    [pd.Series, pd.Series], # y_true, y_pred
    t.Dict[t.Hashable, t.Union[float, int]] # score for each label
]


CP = t.TypeVar('CP', bound='ClassPerformanceImbalanceCheck')


class ClassPerformanceImbalanceCheck(SingleDatasetBaseCheck):
    """Visualize class imbalance by displaying the difference between class metrics."""

    def __init__(
        self,
        metrics: t.Optional[t.Mapping[str, MetricFunc]] = None
    ):
        """Initialize ClassPerformanceImbalanceCheck check.

        Args:
            metrics: alternative metrics to execute
        """
        super().__init__()
        self.alternative_metrics = metrics

        not_callables = (
            it for it in (self.alternative_metrics or {}).values()
            if not callable(it)
        )

        if (incorrect_argument := next(not_callables, None)):
            raise ValueError(
                f'Expected to receive dict of callables but got {type(incorrect_argument)}!'
            )

    def run(
        self,
        dataset: Dataset,
        model: t.Any # TODO: find more precise type for model
    ) -> CheckResult:
        """Run Check.

        Args:
            dataset (Dataset): a dataset object
            model (BaseEstimator): A scikit-learn-compatible fitted estimator instance

        Returns:
            CheckResult: value is a dictionary in format {'label': {'metric name': <value>, ...}, ...}

        Raises:
            DeepchecksValueError:
                if task type is not binary or multi-class;
                if dataset does not have a label column or the label dtype is unsupported;
                if provided dataset is empty;
        """
        return self._class_performance_imbalance(dataset, model)

    def _class_performance_imbalance(
        self,
        dataset: Dataset,
        model: t.Any # TODO: find more precise type for model
    ) -> CheckResult:
        check_name = type(self).__name__
        expected_model_types = [ModelType.BINARY, ModelType.MULTICLASS]

        dataset.validate_label(check_name)
        Dataset.validate_dataset(dataset, check_name)
        task_type_validation(model, dataset, expected_model_types, check_name)

        features = dataset.features_columns()

        if features is None:
            raise DeepchecksValueError(f'Check {check_name} requires dataset with features!')

        y_true = t.cast(pd.Series, dataset.label_col())
        y_pred = model.predict(features)

        if self.alternative_metrics is not None:
            df = pd.DataFrame.from_dict(self._execute_alternative_metrics(y_true, y_pred))

        else:
            labels = unique_labels(y_true, y_pred)
            df = pd.DataFrame.from_dict({
                name: dict(zip(
                    labels,
                    t.cast(t.Iterable, metric_func(y_true, y_pred, labels=labels))
                ))
                for name, metric_func in self._get_default_metrics().items()
            })

        def display():
            title = (
                'Class Performance Imbalance Check for binary data'
                if len(labels) == 2
                else 'Class Performance Imbalance Check for multi-class data'
            )

            df.transpose().plot.bar(
                title=title,
                backend='matplotlib',
                xlabel='Metrics',
                ylabel='Values'
            )

        return CheckResult(
            value=df.transpose().to_dict(),
            header='Class Performance Imbalance',
            check=type(self),
            display=display
        )

    def _get_default_metrics(self) -> t.Dict[str, t.Callable[..., np.ndarray]]:
        return {
            'Accuracy': partial(recall_score, zero_division=0, average=None),
            'Precision': partial(precision_score, zero_division=0, average=None),
            'Recall': partial(recall_score, zero_division=0, average=None)
        }

    def _execute_alternative_metrics(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
    ) -> t.Dict[str, t.Dict[t.Hashable, t.Union[int, float]]]:
        check_name = type(self).__name__
        metrics = t.cast(t.Mapping[str, MetricFunc], self.alternative_metrics)
        result: t.Dict[str, t.Dict[t.Hashable, t.Union[int, float]]] = {}

        for metric_name, metric_func in metrics.items():
            metric_result = metric_func(y_true, y_pred)

            if not isinstance(metric_result, dict):
                raise DeepchecksValueError(
                    f'Check {check_name} expecting that alternative metrics will return '
                    f"not empty instance of 'Dict[Hasable, float|int]', but got {type(metric_result).__name__}" #pylint: disable=inconsistent-quotes
                )

            if len(metric_result) == 0:
                raise DeepchecksValueError(
                    f'Check {check_name} expecting that alternative metrics will return '
                    "not empty instance of 'Dict[Hasable, float|int]'" #pylint: disable=inconsistent-quotes
                )

            incorrect_values = [v for v in metric_result.values() if not isinstance(v, (int, float))]

            if len(incorrect_values) != 0:
                value_type = type(incorrect_values[0]).__name__
                raise DeepchecksValueError(
                    f'Check {check_name} expecting that alternative metrics will return '
                    f"not empty instance of 'Dict[Hasable, float|int]', but got Dict[Hashable, {value_type}]" #pylint: disable=inconsistent-quotes
                )

            result[metric_name] = metric_result

        return result

    def add_condition_percentage_difference_not_greater_than(self: CP, threshold: float = 0.3) -> CP:
        """Add condition.

        Verifying that relative percentage difference
        between highest-class and lowest-class is not greater than 'threshold'.

        Args:
            threshold: percentage difference threshold

        Returns:
            Self: instance of 'ClassPerformanceImbalanceCheck' or it subtype
        """

        def condition(check_result: t.Dict[str, t.Dict[t.Hashable, float]]) -> ConditionResult:
            data = t.cast(
                t.Dict[str, t.Dict[t.Hashable, float]],
                pd.DataFrame.from_dict(check_result).transpose().to_dict()
            )

            result = defaultdict(dict)

            # For each metric calculate: (highest-class - lowest-class)/highest-class
            for metric_name, classes_values in data.items():
                getval = lambda it: it[1]
                lowest_class_name, min_value = min(classes_values.items(), key=getval)
                highest_class_name, max_value = max(classes_values.items(), key=getval)

                relative_difference = (max_value - min_value) / max_value

                if relative_difference >= threshold:
                    result[metric_name][(lowest_class_name, highest_class_name)] = relative_difference

            if len(result) == 0:
                return ConditionResult(True)

            details = '\n'.join([
                f'Metric: {metric_name}, lowest class: {lowest_class_name}, highest class: {highest_class_name};'
                for metric_name, difference in result.items()
                for ((lowest_class_name, highest_class_name), _) in difference.items()
            ])

            details = (
                'Relative percentage difference between highest and lowest classes is greater '
                f'than {format_percent(threshold)}:\n{details}'
            )

            return ConditionResult(False, details=details)

        return self.add_condition(
            name=f'Relative percentage difference is not greater than {format_percent(threshold)}',
            condition_func=condition
        )
