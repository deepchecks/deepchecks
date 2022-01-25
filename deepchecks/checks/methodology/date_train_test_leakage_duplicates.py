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
"""The date_leakage check module."""
import pandas as pd

from deepchecks.base.check_context import CheckRunContext
from deepchecks import CheckResult, TrainTestBaseCheck, ConditionResult
from deepchecks.utils.strings import format_percent, format_datetime


__all__ = ['DateTrainTestLeakageDuplicates']


class DateTrainTestLeakageDuplicates(TrainTestBaseCheck):
    """Check if test dates are present in train data.

    Parameters
    ----------
    n_to_show : int , default: 5
        Number of common dates to show.
    """

    def __init__(self, n_to_show: int = 5):
        super().__init__()
        self.n_to_show = n_to_show

    def run_logic(self, context: CheckRunContext) -> CheckResult:
        """Run check.

        Returns
        -------
        CheckResult
            value is the ratio of date leakage.
            data is html display of the checks' textual result.

        Raises
        ------
        DeepchecksValueError
            If one of the datasets is not a Dataset instance with an date
        """
        train_dataset = context.train
        test_dataset = context.test

        context.assert_datetime_exists()
        train_date = train_dataset.datetime_col
        val_date = test_dataset.datetime_col

        date_intersection = tuple(set(train_date).intersection(val_date))

        if len(date_intersection) > 0:
            leakage_ratio = len(date_intersection) / test_dataset.n_samples
            text = f'{format_percent(leakage_ratio)} of test data dates appear in training data'
            table = pd.DataFrame(
                [[list(format_datetime(it) for it in date_intersection[:self.n_to_show])]],
                index=['Sample of test dates in train:']
            )
            display = [text, table]
            return_value = leakage_ratio
        else:
            display = None
            return_value = 0

        return CheckResult(value=return_value, header='Date Train-Test Leakage (duplicates)', display=display)

    def add_condition_leakage_ratio_not_greater_than(self, max_ratio: float = 0):
        """Add condition - require leakage ratio to not surpass max_ratio.

        Parameters
        ----------
        max_ratio : float , default: 0
            Maximum ratio of leakage.
        """
        def max_ratio_condition(result: float) -> ConditionResult:
            if result > max_ratio:
                return ConditionResult(False, f'Found {format_percent(result)} leaked dates')
            else:
                return ConditionResult(True)

        return self.add_condition(f'Date leakage ratio is not greater than {format_percent(max_ratio)}',
                                  max_ratio_condition)
