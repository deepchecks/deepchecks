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
from deepchecks import CheckResult, Dataset, TrainTestBaseCheck, ConditionResult
from deepchecks.utils.strings import format_percent


__all__ = ['DateTrainTestLeakageOverlap']


class DateTrainTestLeakageOverlap(TrainTestBaseCheck):
    """Check test data that is dated earlier than latest date in train."""

    def run(self, train_dataset: Dataset, test_dataset: Dataset, model=None) -> CheckResult:
        """Run check.

        Arguments:
            train_dataset (Dataset): The training dataset object. Must contain an date column.
            test_dataset (Dataset): The test dataset object. Must contain an date column.
            model: any = None - not used in the check

        Returns:
           CheckResult:
                - value is the ratio of date leakage.
                - data is html display of the checks' textual result.

        Raises:
            DeepchecksValueError: If one of the datasets is not a Dataset instance with an date
        """
        return self._date_train_test_leakage_overlap(train_dataset, test_dataset)

    def _date_train_test_leakage_overlap(self, train_dataset: Dataset, test_dataset: Dataset):
        train_dataset = Dataset.validate_dataset(train_dataset)
        test_dataset = Dataset.validate_dataset(test_dataset)
        train_dataset.validate_date()
        test_dataset.validate_date()

        train_date = train_dataset.date_col
        val_date = test_dataset.date_col

        max_train_date = max(train_date)
        dates_leaked = sum(date <= max_train_date for date in val_date)

        if dates_leaked > 0:
            leakage_ratio = dates_leaked / test_dataset.n_samples
            display = f'{format_percent(leakage_ratio)} of test data dates '\
                      f'before last training data date ({max_train_date})'
            return_value = leakage_ratio
        else:
            display = None
            return_value = 0

        return CheckResult(value=return_value, header='Date Train-Test Leakage (overlap)', display=display)

    def add_condition_leakage_ratio_not_greater_than(self, max_ratio: float = 0):
        """Add condition - require leakage ratio to not surpass max_ratio.

        Args:
            max_ratio (int): Maximum ratio of leakage.
        """
        def max_ratio_condition(result: float) -> ConditionResult:
            if result > max_ratio:
                return ConditionResult(False, f'Found {format_percent(result)} leaked dates')
            else:
                return ConditionResult(True)

        return self.add_condition(f'Date leakage ratio is not greater than {format_percent(max_ratio)}',
                                  max_ratio_condition)
