"""The regression_error_distribution check module."""
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import kurtosis, kurtosistest
from sklearn.base import BaseEstimator

from deepchecks import CheckResult, Dataset, SingleDatasetBaseCheck, ConditionResult
from deepchecks.utils.metrics import ModelType, task_type_validation
from deepchecks.utils.strings import format_number

__all__ = ['RegressionErrorDistribution']


class RegressionErrorDistribution(SingleDatasetBaseCheck):
    """Check regresstion error distribution.

    Args:
        n_top_samples (int): amount of samples to show which are of Largest under / over estimation errors.
        alternative (str): Defines the alternative hypothesis to calculate the p value
            ‘two-sided’: the kurtosis of the distribution underlying the sample is different from that of
                         the normal distribution
            ‘less’: the kurtosis of the distribution underlying the sample is less than that of
                    the normal distribution
            ‘greater’: the kurtosis of the distribution underlying the sample is greater than that of
                        the normal distribution
            (taken from: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kurtosistest.html)
            (Only in python 3.7>=)
    """

    def __init__(self, n_top_samples: int = 3, alternative: str = 'two-sided'):
        super().__init__()
        self.n_top_samples = n_top_samples
        self.alternative = alternative

    def run(self, dataset: Dataset, model: BaseEstimator) -> CheckResult:
        """Run check.

        Arguments:
            dataset (Dataset): A dataset object.
            model (BaseEstimator): A scikit-learn-compatible fitted estimator instance

        Returns:
           CheckResult:
                - value is the kurtosis value (Fisher’s definition (normal ==> 0.0)) and the p score of it.
                - display is histogram of error distirbution and the largest prediction errors.

        Raises:
            DeepchecksValueError: If the object is not a Dataset instance with a label
        """
        return self._regression_error_distribution(dataset, model)

    def _regression_error_distribution(self, dataset: Dataset, model: BaseEstimator):
        check_name = self.__class__.__name__
        Dataset.validate_dataset(dataset, check_name)
        dataset.validate_label(check_name)
        task_type_validation(model, dataset, [ModelType.REGRESSION], check_name)

        y_test = dataset.label_col
        y_pred = model.predict(dataset.features_columns)

        diff = y_test - y_pred
        kurtosis_value = kurtosis(diff)
        # alternative doesn't work on pytho3.6
        try:
            kurtosis_pvalue = kurtosistest(diff, alternative = self.alternative).pvalue
        except TypeError:
            kurtosis_pvalue = kurtosistest(diff).pvalue

        n_largest_diff = diff.nlargest(self.n_top_samples)
        n_largest_diff.name= n_largest_diff.name + '_pred_diff'
        n_largest = pd.concat([dataset.data.loc[n_largest_diff.index], n_largest_diff], axis=1)

        n_smallest_diff = diff.nsmallest(self.n_top_samples)
        n_smallest_diff.name= n_smallest_diff.name + '_pred_diff'
        n_smallest = pd.concat([dataset.data.loc[n_smallest_diff.index], n_smallest_diff], axis=1)

        def display_hist():
            diff = y_test - y_pred
            diff.hist(bins = 40)
            plt.title('Histogram of prediction errors')
            plt.xlabel(f'{dataset.label_name} prediction error')
            plt.ylabel('Count')

        display = [display_hist, 'Largest over estimation errors:', n_largest,
                   'Largest under estimation errors:', n_smallest,]
        return CheckResult(value={'kurtosis': kurtosis_value, 'pvalue': kurtosis_pvalue}, display=display)

    def add_condition_p_value_not_less_than(self, p_value_threshold: float = 0.0001):
        """Add condition - require min p value allowed to be not less than p_value_threshold.

        Args:
            p_value_threshold (float): Minimal p-value to pass the statistical test determining
              if the kurtosis of the distribution is different from that of the normal distribution (0-1).
        """
        def min_p_value_condition(result: float) -> ConditionResult:
            pvalue = result['pvalue']
            if pvalue < p_value_threshold:
                return ConditionResult(False, f'p value: {format_number(pvalue, 5)}')
            else:
                return ConditionResult(True)

        return self.add_condition(f'P value not less than {format_number(p_value_threshold, 5)}',
                                  min_p_value_condition)
