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


class RegressionErrorDistribution(SingleDatasetBaseCheck):
    """Calculate MSE and kurtosis, display an error histogram and most extreme prediction errors.

    Args:
        n_extreme (int): amount of samples to show which are of Largest under / over estimation errors.
    """

    def __init__(self, n_extreme: int = 3):
        super().__init__()
        self.n_extreme = n_extreme

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
        kurtosis_value = kurtosis(diff)

        n_largest_diff = diff.nlargest(self.n_extreme)
        n_largest_diff.name= n_largest_diff.name + '_pred_diff'
        n_largest = pd.concat([dataset.data.loc[n_largest_diff.index], n_largest_diff], axis=1)

        n_smallest_diff = diff.nsmallest(self.n_extreme)
        n_smallest_diff.name= n_smallest_diff.name + '_pred_diff'
        n_smallest = pd.concat([dataset.data.loc[n_smallest_diff.index], n_smallest_diff], axis=1)

        def display_hist():
            diff = y_test - y_pred
            diff.hist(bins = 40)
            plt.title('Histogram of prediction errors')
            plt.xlabel(f'{dataset.label_name()} prediction error')
            plt.ylabel('Frequency')

        display = [display_hist, 'Largest over estimation errors:', n_largest,
                   'Largest under estimation errors:', n_smallest,]
        return CheckResult(value={'mse': mse, 'kurtosis': kurtosis_value}, check=self.__class__, display=display)

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
