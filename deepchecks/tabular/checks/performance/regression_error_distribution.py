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
import plotly.express as px
import pandas as pd
from scipy.stats import kurtosis

from deepchecks.core import CheckResult, ConditionResult, ConditionCategory
from deepchecks.tabular import Context, SingleDatasetCheck
from deepchecks.utils.strings import format_number


__all__ = ['RegressionErrorDistribution']


class RegressionErrorDistribution(SingleDatasetCheck):
    """Check regression error distribution.

    The check shows the distribution of the regression error, and enables to set conditions on the distribution
    kurtosis. Kurtosis is a measure of the shape of the distribution, helping us understand if the distribution
    is significantly "wider" from the normal distribution, which may imply a certain cause of error deforming the
    normal shape.

    Parameters
    ----------
    n_top_samples : int , default: 3
        amount of samples to show which have the largest under / over estimation errors.
    n_bins : int , default: 40
        number of bins to use for the histogram.
    """

    def __init__(self, n_top_samples: int = 3, n_bins: int = 40):
        super().__init__()
        self.n_top_samples = n_top_samples
        self.n_bins = n_bins

    def run_logic(self, context: Context, dataset_type: str = 'train') -> CheckResult:
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
        if dataset_type == 'train':
            dataset = context.train
        else:
            dataset = context.test

        context.assert_regression_task()
        model = context.model
        x_test = dataset.features_columns
        y_test = dataset.label_col

        y_pred = model.predict(x_test)
        y_pred = pd.Series(y_pred, name='predicted ' + str(dataset.label_name), index=y_test.index)

        diff = y_test - y_pred
        kurtosis_value = kurtosis(diff)

        n_largest_diff = diff.nlargest(self.n_top_samples)
        n_largest_diff.name = str(dataset.label_name) + ' Prediction Difference'
        n_largest = pd.concat([dataset.data.loc[n_largest_diff.index], y_pred.loc[n_largest_diff.index],
                               n_largest_diff], axis=1)

        n_smallest_diff = diff.nsmallest(self.n_top_samples)
        n_smallest_diff.name = str(dataset.label_name) + ' Prediction Difference'
        n_smallest = pd.concat([dataset.data.loc[n_smallest_diff.index], y_pred.loc[n_smallest_diff.index],
                                n_smallest_diff], axis=1)

        display = [
            px.histogram(
                x=diff.values,
                nbins=self.n_bins,
                title='Histogram of prediction errors',
                labels={'x': f'{dataset.label_name} prediction error', 'y': 'Count'},
                width=700, height=500
            ),
            'Largest over estimation errors:', n_largest,
            'Largest under estimation errors:', n_smallest
        ]

        return CheckResult(value=kurtosis_value, display=display)

    def add_condition_kurtosis_not_less_than(self, min_kurtosis: float = -0.1):
        """Add condition - require min kurtosis value to be not less than min_kurtosis.

        Parameters
        ----------
        min_kurtosis : float , default: -0.1
            Minimal kurtosis.
        """
        def min_kurtosis_condition(result: float) -> ConditionResult:
            if result < min_kurtosis:
                return ConditionResult(False, f'Found kurtosis below threshold: {format_number(result, 5)}',
                                       category=ConditionCategory.WARN)
            else:
                return ConditionResult(True)

        return self.add_condition(f'Kurtosis value is not less than {format_number(min_kurtosis, 5)}',
                                  min_kurtosis_condition)
