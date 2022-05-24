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
"""The date_leakage check module."""
from deepchecks.core import CheckResult, ConditionCategory, ConditionResult
from deepchecks.tabular import Context, TrainTestCheck
from deepchecks.utils.strings import format_datetime, format_percent

__all__ = ['DateTrainTestLeakageOverlap']


class DateTrainTestLeakageOverlap(TrainTestCheck):
    """Check test data that is dated earlier than latest date in train."""

    def run_logic(self, context: Context) -> CheckResult:
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

        train_dataset.assert_datetime()
        train_date = train_dataset.datetime_col
        val_date = test_dataset.datetime_col

        max_train_date = max(train_date)
        dates_leaked = sum(date <= max_train_date for date in val_date)

        if dates_leaked > 0:
            leakage_ratio = dates_leaked / test_dataset.n_samples
            display = f'{format_percent(leakage_ratio)} of test data dates '\
                      f'before last training data date ({format_datetime(max_train_date)})'
            return_value = leakage_ratio
        else:
            display = None
            return_value = 0

        return CheckResult(value=return_value, header='Date Train-Test Leakage (overlap)', display=display)

    def add_condition_leakage_ratio_not_greater_than(self, max_ratio: float = 0):
        """Add condition - require leakage ratio to not surpass max_ratio.

        Parameters
        ----------
        max_ratio : float , default: 0
            Maximum ratio of leakage.
        """
        def max_ratio_condition(result: float) -> ConditionResult:
            if result == 0:
                return ConditionResult(ConditionCategory.PASS, 'No leaked dates found')
            details = f'Found {format_percent(result)} leaked dates'
            category = ConditionCategory.FAIL if result > max_ratio else ConditionCategory.PASS
            return ConditionResult(category, details)

        return self.add_condition(f'Date leakage ratio is not greater than {format_percent(max_ratio)}',
                                  max_ratio_condition)
