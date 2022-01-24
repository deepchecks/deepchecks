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
from sklearn.metrics import mean_squared_error

from deepchecks.base.check_context import CheckRunContext
from deepchecks import CheckResult, SingleDatasetBaseCheck, ConditionResult
from deepchecks.utils.strings import format_number


__all__ = ['RegressionSystematicError']


class RegressionSystematicError(SingleDatasetBaseCheck):
    """Check the regression systematic error."""

    def run_logic(self, context: CheckRunContext, dataset_type: str = 'train') -> CheckResult:
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
        if dataset_type == 'train':
            dataset = context.train
        else:
            dataset = context.test

        context.assert_regression_task()
        y_test = dataset.data[context.label_name]
        x_test = dataset.data[context.features]
        y_pred = context.model.predict(x_test)

        rmse = mean_squared_error(y_test, y_pred, squared=False)
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

        Parameters
        ----------
        max_ratio : float , default: 0.01
            Maximum ratio
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
