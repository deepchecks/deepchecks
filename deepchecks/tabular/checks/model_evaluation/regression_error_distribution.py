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
"""The regression_error_distribution check module."""
from typing import Dict

import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import kurtosis
from sklearn.metrics import mean_squared_error

from deepchecks.core import CheckResult, ConditionCategory, ConditionResult
from deepchecks.tabular import Context, SingleDatasetCheck
from deepchecks.utils.strings import format_number

__all__ = ['RegressionErrorDistribution']


class RegressionErrorDistribution(SingleDatasetCheck):
    """Check for systematic error and abnormal shape in the regression error distribution.

    The check shows the distribution of the regression error, and enables to set conditions on two
    of the distribution parameters: Systematic error and Kurtosis value.
    Kurtosis is a measure of the shape of the distribution, helping us understand if the distribution
    is significantly "wider" from a normal distribution.
    Systematic error, otherwise known as the error bias, is the mean prediction error of the model.

    Parameters
    ----------
    n_top_samples : int , default: 3
        amount of samples to show which have the largest under / over estimation errors.
    n_bins : int , default: 40
        number of bins to use for the histogram.
    n_samples : int , default: 1_000_000
        number of samples to use for this check.
    random_state : int, default: 42
        random seed for all check internals.
    """

    def __init__(self,
                 n_top_samples: int = 3,
                 n_bins: int = 40,
                 n_samples: int = 1_000_000,
                 random_state: int = 42,
                 **kwargs):
        super().__init__(**kwargs)
        self.n_top_samples = n_top_samples
        self.n_bins = n_bins
        self.n_samples = n_samples
        self.random_state = random_state

    def run_logic(self, context: Context, dataset_kind) -> CheckResult:
        """Run check.

        Returns
        -------
        CheckResult
            value is the kurtosis value (Fisher’s definition (normal ==> 0.0)).
            display is histogram of error distribution and the largest prediction errors.

        Raises
        ------
        DeepchecksValueError
            If the object is not a Dataset instance with a label
        """
        dataset = context.get_data_by_kind(dataset_kind).sample(self.n_samples, random_state=self.random_state)
        context.assert_regression_task()
        model = context.model
        y_test = dataset.label_col

        y_pred = model.predict(dataset.features_columns)
        y_pred = pd.Series(y_pred, name='predicted ' + str(dataset.label_name), index=y_test.index)

        diff = y_test - y_pred
        kurtosis_value = kurtosis(diff)

        if context.with_display:
            n_largest_diff = diff.nlargest(self.n_top_samples)
            n_largest_diff.name = str(dataset.label_name) + ' Prediction Difference'
            n_largest = pd.concat([dataset.data.loc[n_largest_diff.index], y_pred.loc[n_largest_diff.index],
                                   n_largest_diff], axis=1)

            n_smallest_diff = diff.nsmallest(self.n_top_samples)
            n_smallest_diff.name = str(dataset.label_name) + ' Prediction Difference'
            n_smallest = pd.concat([dataset.data.loc[n_smallest_diff.index], y_pred.loc[n_smallest_diff.index],
                                    n_smallest_diff], axis=1)
            fig = px.histogram(
                x=diff.values,
                nbins=self.n_bins,
                title='Regression Error Distribution',
                labels={'x': f'{dataset.label_name} prediction error', 'y': 'Count'},
                height=500
            )

            fig.add_vline(x=np.median(diff), line_dash='dash', line_color='purple', annotation_text='median',
                          annotation_position='top left' if np.median(diff) < np.mean(diff) else 'top right')
            fig.add_vline(x=np.mean(diff), line_dash='dot', line_color='purple', annotation_text='mean',
                          annotation_position='top right' if np.median(diff) < np.mean(diff) else 'top left')

            display = [
                fig,
                'Largest over estimation errors:', n_largest,
                'Largest under estimation errors:', n_smallest
            ]
        else:
            display = None

        results = {
            'Mean Prediction Error': np.mean(diff),
            'Median Prediction Error': np.median(diff),
            'Kurtosis Value': kurtosis_value,
            'RMSE': mean_squared_error(y_test, y_pred, squared=False)
        }

        return CheckResult(value=results, display=display)

    def add_condition_kurtosis_greater_than(self, threshold: float = -0.1):
        """Add condition - require kurtosis value to be greater than the provided threshold.

        Kurtosis is a measure of the shape of the distribution, helping us understand if the distribution
        is significantly "wider" from a normal distribution. A lower value indicates a "wider" distribution.
        Parameters
        ----------
        threshold : float , default: -0.1
            Minimal threshold for kurtosis value.
        """

        def min_kurtosis_condition(result: Dict[str, float]) -> ConditionResult:
            details = f'Found kurtosis value of {format_number(result["Kurtosis Value"], 5)}'
            category = ConditionCategory.PASS if result['Kurtosis Value'] > threshold else ConditionCategory.WARN
            return ConditionResult(category, details)

        return self.add_condition(f'Kurtosis value higher than {format_number(threshold, 5)}',
                                  min_kurtosis_condition)

    def add_condition_systematic_error_ratio_to_rmse_less_than(self, max_ratio: float = 0.01):
        """Add condition - require systematic error (mean error) lower than (max_ratio * RMSE).

        Parameters
        ----------
        max_ratio : float , default: 0.01
            Maximum ratio allowed between the mean error and the rmse value
        """

        def max_bias_condition(result: Dict[str, float]) -> ConditionResult:
            ratio = abs(result['Mean Prediction Error']) / result['RMSE']
            details = f'Found systematic error to rmse ratio of {format_number(ratio)}'
            category = ConditionCategory.PASS if ratio < max_ratio else ConditionCategory.FAIL
            return ConditionResult(category, details)

        return self.add_condition(f'Systematic error ratio lower than {format_number(max_ratio)}',
                                  max_bias_condition)
