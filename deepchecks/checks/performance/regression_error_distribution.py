# ----------------------------------------------------------------------------
# Copyright (C) 2021 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""The regression_error_distribution check module."""
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import kurtosis
from sklearn.base import BaseEstimator

from deepchecks import CheckResult, Dataset, SingleDatasetBaseCheck, ConditionResult
from deepchecks.utils.metrics import ModelType, task_type_validation
from deepchecks.utils.strings import format_number

__all__ = ['RegressionErrorDistribution']


class RegressionErrorDistribution(SingleDatasetBaseCheck):
    """Check regresstion error distribution.

    Args:
        n_top_samples (int): amount of samples to show which are of Largest under / over estimation errors.
    """

    def __init__(self, n_top_samples: int = 3):
        super().__init__()
        self.n_top_samples = n_top_samples

    def run(self, dataset: Dataset, model: BaseEstimator) -> CheckResult:
        """Run check.

        Arguments:
            dataset (Dataset): A dataset object.
            model (BaseEstimator): A scikit-learn-compatible fitted estimator instance

        Returns:
           CheckResult:
                - value is the kurtosis value (Fisher’s definition (normal ==> 0.0)).
                - display is histogram of error distirbution and the largest prediction errors.

        Raises:
            DeepchecksValueError: If the object is not a Dataset instance with a label
        """
        return self._regression_error_distribution(dataset, model)

    def _regression_error_distribution(self, dataset: Dataset, model: BaseEstimator):
        Dataset.validate_dataset(dataset)
        dataset.validate_label()
        task_type_validation(model, dataset, [ModelType.REGRESSION])

        y_test = dataset.label_col
        y_pred = model.predict(dataset.features_columns)

        diff = y_test - y_pred
        kurtosis_value = kurtosis(diff)

        n_largest_diff = diff.nlargest(self.n_top_samples)
        n_largest_diff.name= n_largest_diff.name + ' Prediction Difference'
        n_largest = pd.concat([dataset.data.loc[n_largest_diff.index], n_largest_diff], axis=1)

        n_smallest_diff = diff.nsmallest(self.n_top_samples)
        n_smallest_diff.name= n_smallest_diff.name + ' Prediction Difference'
        n_smallest = pd.concat([dataset.data.loc[n_smallest_diff.index], n_smallest_diff], axis=1)

        def display_hist():
            diff = y_test - y_pred
            diff.hist(bins = 40)
            plt.title('Histogram of prediction errors')
            plt.xlabel(f'{dataset.label_name} prediction error')
            plt.ylabel('Count')

        display = [display_hist, 'Largest over estimation errors:', n_largest,
                   'Largest under estimation errors:', n_smallest,]
        return CheckResult(value=kurtosis_value, display=display)

    def add_condition_kurtosis_not_less_than(self, min_kurtosis: float = -0.1):
        """Add condition - require min kurtosis value to be not less than min_kurtosis.

        Args:
            min_kurtosis (float): Minimal kurtosis.
        """
        def min_kurtosis_condition(result: float) -> ConditionResult:
            if result < min_kurtosis:
                return ConditionResult(False, f'kurtosis: {format_number(result, 5)}')
            else:
                return ConditionResult(True)

        return self.add_condition(f'Kurtosis value not less than {format_number(min_kurtosis, 5)}',
                                  min_kurtosis_condition)
