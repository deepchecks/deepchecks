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
"""The regression_error_bias check module."""
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error

from deepchecks import CheckResult, Dataset, SingleDatasetBaseCheck, ConditionResult
from deepchecks.utils.metrics import ModelType, task_type_validation
from deepchecks.utils.strings import format_number

__all__ = ['RegressionBias']


class RegressionBias(SingleDatasetBaseCheck):
    """Check the regression bias."""

    def run(self, dataset: Dataset, model: BaseEstimator) -> CheckResult:
        """Run check.

        Arguments:
            dataset (Dataset): A dataset object.
            model (BaseEstimator): A scikit-learn-compatible fitted estimator instance
        Returns:
           CheckResult:
                - value is a dict with rmse and mean prediction error.
                - display is box plot of the prediction errorד.
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

        rmse = mean_squared_error(dataset.label_col, y_pred, squared=False)
        diff = y_test - y_pred
        diff_mean = diff.mean()

        def display():
            red_square = dict(markerfacecolor='r', marker='s')
            _, ax = plt.subplots()
            ax.set_title('Prediction Errors')
            ax.boxplot(diff, vert=False, flierprops=red_square)

        return CheckResult(value={'rmse': rmse, 'mean_error': diff_mean}, check=self.__class__, display=display)

    def add_condition_bias_ratio_not_greater_than(self, max_ratio: float = 0.01):
        """Add condition - require the absolute mean error to be not greater than (max_ratio * RMSE).

        Args:
            max_kurtosis (float): Maximum absolute kurtosis value
        """
        def max_bias_condition(result: float) -> ConditionResult:
            rmse = result['rmse']
            mean_error = result['mean_error']
            if abs(mean_error) > max_ratio * rmse:
                return ConditionResult(False,
                                      f'mean error: {format_number(mean_error, 5)}, RMSE: {format_number(rmse)}')
            else:
                return ConditionResult(True)

        return self.add_condition(f'Bias ratio is not greater than {format_number(max_ratio)}',
                                  max_bias_condition)
