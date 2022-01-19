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
import plotly.graph_objects as go
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error

from deepchecks import CheckResult, Dataset, SingleDatasetBaseCheck, ConditionResult
from deepchecks.utils.metrics import ModelType
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
        dataset = Dataset.ensure_not_empty_dataset(dataset)
        y_test = self._dataset_has_label(dataset)
        x_test = self._dataset_has_features(dataset)
        self._verify_model_type(model, dataset, [ModelType.REGRESSION])

        y_pred = model.predict(x_test)

        rmse = mean_squared_error(dataset.label_col, y_pred, squared=False)
        diff = y_test - y_pred
        diff_mean = diff.mean()

        fig = (
            go.Figure()
            .add_trace(go.Box(
                x=diff,
                orientation='h',
                name='Model prediction error',
                hoverinfo='x',
                boxmean=True))
            .update_layout(
                title_text='Box plot of the model prediction error',
                width=800,
                height=500)
        )

        display = [
            'Non-zero mean of the error distribution indicated the presents '
            'of systematic error in model predictions',
            fig
        ]

        return CheckResult(value={'rmse': rmse, 'mean_error': diff_mean}, display=display)

    def add_condition_systematic_error_ratio_to_rmse_not_greater_than(self, max_ratio: float = 0.01):
        """Add condition - require the absolute mean systematic error to be not greater than (max_ratio * RMSE).

        Args:
            max_ratio (float): Maximum ratio
        """
        def max_bias_condition(result: dict) -> ConditionResult:
            rmse = result['rmse']
            mean_error = result['mean_error']
            ratio = abs(mean_error) / rmse
            if ratio > max_ratio:
                return ConditionResult(False, f'Found bias ratio above threshold: {format_number(ratio)}')
            else:
                return ConditionResult(True)

        return self.add_condition(f'Bias ratio is not greater than {format_number(max_ratio)}',
                                  max_bias_condition)
