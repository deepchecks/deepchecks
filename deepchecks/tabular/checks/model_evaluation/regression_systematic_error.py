# ----------------------------------------------------------------------------
# Copyright (C) 2021-2022 Deepchecks (https://www.deepchecks.com)
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
from sklearn.metrics import mean_squared_error

from deepchecks.core import CheckResult, ConditionResult
from deepchecks.core.condition import ConditionCategory
from deepchecks.tabular import Context, SingleDatasetCheck
from deepchecks.utils.strings import format_number

__all__ = ['RegressionSystematicError']


class RegressionSystematicError(SingleDatasetCheck):
    """Check the regression systematic error."""

    def run_logic(self, context: Context, dataset_kind) -> CheckResult:
        """Run check.

        Returns
        -------
        CheckResult
            value is a dict with rmse and mean prediction error.
            display is box plot of the prediction error.

        Raises
        ------
        DeepchecksValueError
            If the object is not a Dataset instance with a label.
        """
        dataset = context.get_data_by_kind(dataset_kind)
        context.assert_regression_task()
        y_test = dataset.label_col
        x_test = dataset.features_columns
        y_pred = context.model.predict(x_test)

        rmse = mean_squared_error(y_test, y_pred, squared=False)
        diff = y_test - y_pred
        diff_mean = diff.mean()

        if context.with_display:
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
                    height=500
                )
            )

            display = [
                'Non-zero mean of the error distribution indicated the presents '
                'of systematic error in model predictions',
                fig
            ]
        else:
            display = None

        return CheckResult(value={'rmse': rmse, 'mean_error': diff_mean}, display=display)

    def add_condition_systematic_error_ratio_to_rmse_less_than(self, max_ratio: float = 0.01):
        """Add condition - require the absolute mean systematic error is less than (max_ratio * RMSE).

        Parameters
        ----------
        max_ratio : float , default: 0.01
            Maximum ratio
        """
        def max_bias_condition(result: dict) -> ConditionResult:
            rmse = result['rmse']
            mean_error = result['mean_error']
            ratio = abs(mean_error) / rmse
            details = f'Found bias ratio {format_number(ratio)}'
            category = ConditionCategory.PASS if ratio < max_ratio else ConditionCategory.FAIL
            return ConditionResult(category, details)

        return self.add_condition(f'Bias ratio is less than {format_number(max_ratio)}',
                                  max_bias_condition)
