"""The train_validation_difference_overfit check module."""
import typing as t

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from deepchecks.utils import model_type_validation, DeepchecksValueError
from deepchecks.metric_utils import get_metrics_list
from deepchecks import (
    Dataset,
    CheckResult,
    TrainTestBaseCheck,
    ConditionResult,
    ConditionCategory
)


__all__ = ['TrainTestDifferenceOverfit']



TD = t.TypeVar('TD', bound='TrainTestDifferenceOverfit')


class TrainTestDifferenceOverfit(TrainTestBaseCheck):
    """Visualize overfit by displaying the difference between model metrics on train and on test data.

    The check would display the selected metrics for the training and test data, helping the user visualize
    the difference in performance between the two datasets. If no alternative_metrics are supplied, the check would
    use a list of default metrics. If they are supplied, alternative_metrics must be a dictionary, with the keys
    being metric names and the values being either a name of an sklearn scoring function
    (https://scikit-learn.org/stable/modules/model_evaluation.html#scoring) or an sklearn scoring function.
    """

    def __init__(
        self,
        alternative_metrics: t.Dict[str, t.Callable[[object, pd.DataFrame, str], float]] = None
    ):
        """
        Initialize the TrainTestDifferenceOverfit check.

        Args:
            alternative_metrics (Dict[str, Callable]): An optional dictionary of metric name to scorer functions
        """
        super().__init__()
        self.alternative_metrics = alternative_metrics

    def run(self, train_dataset: Dataset, test_dataset: Dataset, model=None) -> CheckResult:
        """Run check.

        Args:
            train_dataset (Dataset): The training dataset object. Must contain a label column.
            test_dataset (Dataset): The test dataset object. Must contain a label column.
            model: A scikit-learn-compatible fitted estimator instance

        Returns:
            CheckResult:
                value is a dataframe with metrics as indexes, and scores per training and test in the columns.
                data is a bar graph of the metrics for training and test data.

        Raises:
            DeepchecksValueError: If either of the dataset objects are not a Dataset instance with a label
        """
        return self._train_test_difference_overfit(train_dataset, test_dataset, model)

    def _train_test_difference_overfit(self, train_dataset: Dataset, test_dataset: Dataset, model,
                                       ) -> CheckResult:
        # Validate parameters
        func_name = self.__class__.__name__
        Dataset.validate_dataset(train_dataset, func_name)
        Dataset.validate_dataset(test_dataset, func_name)
        train_dataset.validate_label(func_name)
        test_dataset.validate_label(func_name)
        train_dataset.validate_shared_label(test_dataset, func_name)
        train_dataset.validate_shared_features(test_dataset, func_name)
        model_type_validation(model)

        metrics = get_metrics_list(model, train_dataset, self.alternative_metrics)

        train_metrics = {key: scorer(model, train_dataset.data[train_dataset.features()], train_dataset.label_col())
                         for key, scorer in metrics.items()}

        test_metrics = {key: scorer(model, test_dataset.data[test_dataset.features()],
                                    test_dataset.label_col())
                        for key, scorer in metrics.items()}

        res_df = pd.DataFrame.from_dict({'Training Metrics': train_metrics,
                                         'Test Metrics': test_metrics})

        def plot_overfit():
            width = 0.20
            my_cmap = plt.cm.get_cmap('Set2')
            indices = np.arange(len(res_df.index))

            colors = my_cmap(range(len(res_df.columns)))
            plt.bar(indices, res_df['Training Metrics'].values.flatten(), width=width, color=colors[0])
            plt.bar(indices + width, res_df['Test Metrics'].values.flatten(), width=width, color=colors[1])
            plt.ylabel('Metrics')
            plt.xticks(ticks=indices + width / 2., labels=res_df.index)
            plt.xticks(rotation=30)
            plt.legend(res_df.columns, loc='upper right', bbox_to_anchor=(1.45, 1.02))

        res = res_df.apply(lambda x: x[1] - x[0], axis=1)
        res.index = res.index.to_series().apply(lambda x: x + ' - Difference between Training data and Test data')

        return CheckResult(res, check=self.__class__, header='Train Test Difference Overfit',
                           display=[plot_overfit])

    def _condition_factory(
        self,
        var: float,
        metrics: t.Union[str, t.Sequence[str], None],
        category: ConditionCategory,
        failure_message: str,
        operation_func: t.Callable[[pd.Series, pd.Series], pd.Series],
        condition_func: t.Callable[[pd.Series, float], pd.Series]
    ) -> t.Callable[[pd.DataFrame], ConditionResult]:
        if metrics is not None and len(metrics) == 0:
            raise DeepchecksValueError("`metrics` names list (name string) cannot be empty!") # pylint: disable=inconsistent-quotes
        elif isinstance(metrics, str):
            metrics = [metrics]
        elif isinstance(metrics, t.Sequence):
            metrics = list(metrics)
        else:
            metrics = None

        def condition(df: pd.DataFrame) -> ConditionResult:
            assert isinstance(metrics, (list, type(None)))
            calculated_metric_names = t.cast(t.List[str], list(df['Training Metrics'].index))
            provided_metric_names = metrics if metrics is not None else calculated_metric_names
            metric_names_diff = set(provided_metric_names).difference(set(calculated_metric_names))

            if len(metric_names_diff) != 0:
                raise DeepchecksValueError(f'Unknown metrics - {metric_names_diff}')

            training_metrics = df['Training Metrics'][provided_metric_names]
            test_metrics = df['Test Metrics'][provided_metric_names]
            operation_result = operation_func(training_metrics, test_metrics)
            condition_result = operation_result[condition_func(operation_result, var)]
            passed = len(condition_result) == 0

            details_vars = {
                'var': var,
                'all_metric_values': ';'.join([f'{k}={v}' for k, v in operation_result.to_dict().items()]),
                'failed_metric_values': ';'.join([f'{k}={v}' for k, v in condition_result.to_dict().items()])
            }

            return ConditionResult(
                is_pass=passed,
                category=category,
                details=(
                    failure_message.format(**details_vars)
                    if not passed
                    else ''
                )
            )

        return condition

    def add_condition_train_is_lower_by(
        self: TD,
        var: float,
        *,
        metrics: t.Union[str, t.Sequence[str], None] = None,
        category = ConditionCategory.FAIL,
        failure_message = 'Condition failed', #TODO: add meaningful message
        name = '' #TODO: add meaningful default name
    ) -> TD:
        """
        Add condition that will check that metric value is not lower than x.

        If `metrics` variable was not passed then this condition will be applied to
        each calculated metric.

        Next variables are available for use within messages templates:
            - var (float)
            - all_metric_values (str)
            - failed_metric_values (str) - metric values that did not pass condition

        Args:
            var
            metrics: list of metric names to which condition should be applied
            category: condition category
            failure_message: condition message in case of failure
            name: condition name

        Raises:
            DeepchecksValueError: if `metrics` is empty list or empty str

        Condition Raises:
            DeepchecksValueError: if `metrics` contains unknown metric name
        """
        return self.add_condition(
            name=name,
            condition_func=self._condition_factory(
                var,
                metrics,
                category,
                failure_message,
                operation_func=lambda trainin_metrics, test_metrics: trainin_metrics - test_metrics,
                condition_func=lambda metrics_difference, var: metrics_difference >= var
            )
        )

    def add_condition_train_is_lower_by_factor_of(
        self: TD,
        var: float,
        *,
        metrics: t.Union[str, t.Sequence[str], None] = None,
        category = ConditionCategory.FAIL,
        failure_message = 'Condition failed', #TODO: add meaningful default message
        name = '' #TODO: add meaningful default name
    ) -> TD:
        """
        Add condition that will check that metric value is not lower by factor of x.

        If `metrics` variable was not passed then this condition will be applied to
        each calculated metric.

        Next variables are available for use within messages templates:
            - var (float)
            - all_metric_values (str)
            - failed_metric_values (str) - metric values that did not pass condition

        Args:
            var
            metrics: list of metric names to which condition should be applied
            category: condition category
            failure_message: condition message in case of failure
            name: condition name

        Raises:
            DeepchecksValueError: if `metrics` is empty list or empty str

        Condition Raises:
            DeepchecksValueError: if `metrics` contains unknown metric name
        """
        return self.add_condition(
            name=name,
            condition_func=self._condition_factory(
                var,
                metrics,
                category,
                failure_message,
                operation_func=lambda trainin_metrics, test_metrics: trainin_metrics / test_metrics,
                condition_func=lambda metrics_ratio, var: metrics_ratio >= var
            )
        )
