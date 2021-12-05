"""The regression_error_distribution check module."""
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import kurtosis
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error

from deepchecks import CheckResult, Dataset, SingleDatasetBaseCheck, ConditionResult
from deepchecks.utils.metrics import ModelType, task_type_validation
from deepchecks.utils.strings import format_number

__all__ = ['RegressionErrorDistribution']


class RegressionBias(SingleDatasetBaseCheck):
    """Calculate MSE and the avarage prediction error"""

    def run(self, dataset: Dataset, model: BaseEstimator) -> CheckResult:
        """Run check.
        Arguments:
            dataset (Dataset): A dataset object.
            model (BaseEstimator): A scikit-learn-compatible fitted estimator instance
        Returns:
           CheckResult:
                - value is a dict with mse and kurtosis values.
                - display is histogram of error distirbution and the extreme prediction errors.
        Raises:
            DeepchecksValueError: If the object is not a Dataset instance with a label
        """
        return self._regression_error_distribution(dataset, model)

    def _regression_error_distribution(self, dataset: Dataset, model: BaseEstimator):
        check_name = self.__class__.__name__
        Dataset.validate_dataset(dataset, check_name)
        dataset.validate_label(check_name)
        task_type_validation(model, dataset, [ModelType.REGRESSION], check_name)

        y_test = dataset.label_col()
        y_pred = model.predict(dataset.features_columns())

        mse = mean_squared_error(dataset.label_col(), y_pred, squared=False)
        diff = y_test - y_pred
        diff_mean = diff.mean()

        def display_hist():
            red_square = dict(markerfacecolor='r', marker='s')
            _, ax = plt.subplots()
            ax.set_title('Horizontal Boxes')
            ax.boxplot(diff, vert=False, flierprops=red_square)

        display = [display_hist, 'Largest over estimation errors:', n_largest,
                   'Largest under estimation errors:', n_smallest,]
        return CheckResult(value={'mse': mse, 'mean_error': diff_mean}, check=self.__class__, display=display)

    def add_condition_absolute_kurtosis_not_greater_than(self, max_kurtosis: float = 0.1):
        """Add condition - require the absolute kurtosis value to not surpass max_kurtosis.
        Args:
            max_kurtosis (float): Maximum absolute kurtosis value
        """
        def max_kurtosis_condition(result: float) -> ConditionResult:
            kurtosis_value = result['kurtosis']
            if abs(kurtosis_value) > max_kurtosis:
                return ConditionResult(False, f'kurtosis: {format_number(kurtosis_value, 5)}')
            else:
                return ConditionResult(True)

        return self.add_condition(f'Absolute kurtosis value is not greater than {format_number(max_kurtosis, 5)}',
                                  max_kurtosis_condition)
