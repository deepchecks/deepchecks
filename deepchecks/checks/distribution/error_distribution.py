"""The date_leakage check module."""
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
from sklearn.base import BaseEstimator

from deepchecks import CheckResult, Dataset, SingleDatasetBaseCheck, ConditionResult
from deepchecks.utils.metrics import ModelType, task_type_validation
from deepchecks.utils.strings import format_number
from sklearn.metrics import mean_squared_error

__all__ = ['DateTrainTestLeakageDuplicates']


class RegressionErrorDistribution(SingleDatasetBaseCheck):
    """Check if test dates are present in train data."""

    def run(self, dataset: Dataset, model: BaseEstimator) -> CheckResult:
        """Run check.

        Arguments:
            dataset (Dataset): A dataset object.
            model (BaseEstimator): A scikit-learn-compatible fitted estimator instance

        Returns:
           CheckResult:
                - value is the ratio of date leakage.
                - data is html display of the checks' textual result.

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

        def display():
            diff = y_test - y_pred
            diff.hist(bins = 40)
            plt.title('Histogram of prediction errors')
            plt.xlabel(f'{dataset.label_name()} prediction error')
            plt.ylabel('Frequency')

        return CheckResult(value={'mse': mse, 'kurtosis': kurtosis_value}, check=self.__class__, display=display)

    def add_condition_absalute_kurtosis_not_greater_than(self, max_kurtosis: float = 0.1):
        """Add condition - require the absalute kurtosis value to not surpass max_kurtosis.

        Args:
            max_ratio (int): Maximum ratio of leakage.
        """
        def max_ratio_condition(result: float) -> ConditionResult:
            kurtosis_value = result['kurtosis']
            if abs(kurtosis_value) > max_kurtosis:
                return ConditionResult(False, f'kurtosis: {format_number(kurtosis_value)}')
            else:
                return ConditionResult(True)

        return self.add_condition(f'Absalute kurtosis value is not greater than {format_number(max_kurtosis)}',
                                  max_ratio_condition)
