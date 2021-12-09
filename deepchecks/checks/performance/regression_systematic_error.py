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
"""The RegressionSystematicError check module."""
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error

from deepchecks import CheckResult, Dataset, SingleDatasetBaseCheck, ConditionResult
from deepchecks.utils.metrics import ModelType, task_type_validation
from deepchecks.utils.strings import format_number

__all__ = ['RegressionSystematicError']


class RegressionSystematicError(SingleDatasetBaseCheck):
    """Check the regression systematic error."""

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
        Dataset.validate_dataset(dataset)
        dataset.validate_label()
        task_type_validation(model, dataset, [ModelType.REGRESSION])

        y_test = dataset.label_col
        y_pred = model.predict(dataset.features_columns)

        rmse = mean_squared_error(dataset.label_col, y_pred, squared=False)
        diff = y_test - y_pred
        diff_mean = diff.mean()

        def display_box_plot():
            red_square = dict(markerfacecolor='r', marker='s')
            _, ax = plt.subplots()
            ax.set_title('Box plot of the model prediction error')
            ax.boxplot(diff, vert=False, flierprops=red_square)
            ax.axvline(x=diff_mean, linestyle='--')
            ax.annotate(xy=(diff_mean + 0.01, 1.2), text='mean error')

        display = ['Non-zero mean of the error distribution indicated the presents of \
            systematic error in model predictions', display_box_plot]

        return CheckResult(value={'rmse': rmse, 'mean_error': diff_mean}, display=display)

    def add_condition_systematic_error_ratio_to_rmse_not_greater_than(self, max_ratio: float = 0.01):
        """Add condition - require the absolute mean systematic error to be not greater than (max_ratio * RMSE).

        Args:
            max_ratio (float): Maximum ratio
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
