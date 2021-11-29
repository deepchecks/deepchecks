"""The class performance imbalance check."""
import typing as t
from functools import partial
from collections import defaultdict

import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, get_scorer
from sklearn.utils.multiclass import unique_labels as get_unique_labels

from deepchecks import SingleDatasetBaseCheck, Dataset, CheckResult, ConditionResult
from deepchecks.metric_utils import task_type_validation, ModelType
from deepchecks.string_utils import format_percent
from deepchecks.utils import DeepchecksValueError


__all__ = ['ClassPerformanceImbalanceCheck']


MetricFunc = t.Callable[
    [object, pd.DataFrame, pd.Series], # model, features, labels
    t.Dict[t.Hashable, t.Union[float, int]] # score for each label
]

AlternativeMetric = t.Union[str, MetricFunc]


CP = t.TypeVar('CP', bound='ClassPerformanceImbalanceCheck')


class ClassPerformanceImbalanceCheck(SingleDatasetBaseCheck):
    """Visualize class imbalance by displaying the difference between class metrics.

    Args:
        alternative_metrics (Mapping[str, Union[str, Callable]]):
            An optional dictionary of metric name or scorer functions

    Raises:
        ValueError:
            if provided dict of metrics is emtpy;
            if one of the entries of the provided metrics dict contains name of unknown scorer;
            if one of the entries of the provided metrics dict contains not callable value;
    """

    def __init__(
        self,
        alternative_metrics: t.Optional[t.Mapping[str, AlternativeMetric]] = None
    ):
        super().__init__()
        self.alternative_metrics: t.Optional[t.Mapping[str, MetricFunc]] = None

        if alternative_metrics is not None and len(alternative_metrics) == 0:
            raise ValueError('alternative_metrics - expected to receive not empty dict of scorers!')

        elif alternative_metrics is not None:
            self.alternative_metrics = {}

            for name, metric in alternative_metrics.items():
                if isinstance(metric, t.Callable):
                    self.alternative_metrics[name] = metric
                elif isinstance(metric, str):
                    self.alternative_metrics[name] = get_scorer(metric)
                else:
                    raise ValueError(
                        f"alternative_metrics - expected to receive 'Mapping[str, Callable]' but got " #pylint: disable=inconsistent-quotes
                        f"'Mapping[str, {type(metric).__name__}]'!" #pylint: disable=inconsistent-quotes
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
        dataset.validate_features(check_name)
        Dataset.validate_dataset(dataset, check_name)
        task_type_validation(model, dataset, expected_model_types, check_name)

        labels = t.cast(pd.Series, dataset.label_col())
        features = t.cast(pd.DataFrame, dataset.features_columns())

        if self.alternative_metrics is not None:
            df = pd.DataFrame.from_dict(self._execute_alternative_metrics(
                model, features, labels
            ))
        else:
            y_true = labels
            y_pred = model.predict(features)
            unique_labels = get_unique_labels(labels, y_pred)
            df = pd.DataFrame.from_dict({
                name: dict(zip(unique_labels, metric_func(y_true, y_pred, labels=unique_labels))) # type: ignore
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
        model: t.Any,
        features: pd.DataFrame,
        labels: pd.Series,
    ) -> t.Dict[str, t.Dict[t.Hashable, t.Union[int, float]]]:
        check_name = type(self).__name__
        metrics = t.cast(t.Mapping[str, MetricFunc], self.alternative_metrics)
        result: t.Dict[str, t.Dict[t.Hashable, t.Union[int, float]]] = {}

        for metric_name, metric_func in metrics.items():
            metric_result = metric_func(model, features, labels)

            if not isinstance(metric_result, dict):
                raise DeepchecksValueError(
                    f'Check {check_name} expecting that alternative metrics will return '
                    f"not empty instance of 'Mapping[Hashable, float|int]', but got {type(metric_result).__name__}" #pylint: disable=inconsistent-quotes
                )

            if len(metric_result) == 0:
                raise DeepchecksValueError(
                    f'Check {check_name} expecting that alternative metrics will return '
                    "not empty instance of 'Mapping[Hashable, float|int]'" #pylint: disable=inconsistent-quotes
                )

            incorrect_values = [v for v in metric_result.values() if not isinstance(v, (int, float))]

            if len(incorrect_values) != 0:
                value_type = type(incorrect_values[0]).__name__
                raise DeepchecksValueError(
                    f'Check {check_name} expecting that alternative metrics will return '
                    "not empty instance of 'Mapping[Hashable, float|int]', "
                    f"but got 'Mapping[Hashable, {value_type}]'" #pylint: disable=inconsistent-quotes
                )

            result[metric_name] = metric_result

        return result

    def add_condition_ratio_difference_not_greater_than(self: CP, threshold: float = 0.3) -> CP:
        """Add condition.

        Verifying that relative ratio difference
        between highest-class and lowest-class is not greater than 'threshold'.

        Args:
            threshold: ratio difference threshold

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

                relative_difference = abs((min_value - max_value) / max_value)

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
                'Relative ratio difference between highest and lowest classes is greater '
                f'than {format_percent(threshold)}:\n{details}'
            )

            return ConditionResult(False, details=details)

        return self.add_condition(
            name=f'Relative ratio difference is not greater than {format_percent(threshold)}',
            condition_func=condition
        )
