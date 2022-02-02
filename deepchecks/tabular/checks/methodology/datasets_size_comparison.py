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
"""Datasets size comparision check module."""
import typing as t
import pandas as pd

from deepchecks.core import CheckResult, ConditionResult
from deepchecks.tabular import Context, TrainTestCheck


__all__ = ['DatasetsSizeComparison']


T = t.TypeVar('T', bound='DatasetsSizeComparison')


class DatasetsSizeComparison(TrainTestCheck):
    """Verify test dataset size comparing it to the train dataset size."""

    def run_logic(self, context: Context) -> CheckResult:
        """Run check.

        Returns
        -------
        CheckResult
            with value of type pandas.DataFrame.
            Value contains two keys, 'train' - size of the train dataset
            and 'test' - size of the test dataset.

        Raises
        ------
        DeepchecksValueError
            if not dataset instances were provided.
            if datasets are empty.
        """
        train_dataset = context.train
        test_dataset = context.test
        sizes = {'Train': len(train_dataset), 'Test': len(test_dataset)}
        display = pd.DataFrame(sizes, index=['Size'])
        return CheckResult(
            value=sizes,
            display=display
        )

    def add_condition_test_size_not_smaller_than(self: T, value: int = 100) -> T:
        """Add condition verifying that size of the test dataset is not smaller than X.

        Parameters
        ----------
        value : int , default: 100
            minimal allowed test dataset size.

        Returns
        -------
        Self
            current instance of the DatasetsSizeComparison check.
        """
        def condition(check_result: dict) -> ConditionResult:
            return (
                ConditionResult(False, f'Test dataset size is {check_result["Test"]}')
                if check_result['Test'] <= value
                else ConditionResult(True)
            )

        return self.add_condition(
            name=f'Test dataset size is not smaller than {value}',
            condition_func=condition
        )

    def add_condition_test_train_size_ratio_not_smaller_than(self: T, ratio: float = 0.01) -> T:
        """Add condition verifying that test-train size ratio is not smaller than X.

        Parameters
        ----------
        ratio : float , default: 0.01
            minimal allowed test-train ratio.

        Returns
        -------
        Self
            current instance of the DatasetsSizeComparison check.
        """

        def condition(check_result: dict) -> ConditionResult:
            test_train_ratio = check_result['Test'] / check_result['Train']
            if test_train_ratio <= ratio:
                return ConditionResult(False, f'Test-Train size ratio is {test_train_ratio}')
            else:
                return ConditionResult(True)

        return self.add_condition(
            name=f'Test-Train size ratio is not smaller than {ratio}',
            condition_func=condition
        )

    def add_condition_train_dataset_not_smaller_than_test(self: T) -> T:
        """Add condition verifying that train dataset is not smaller than test dataset.

        Returns
        -------
        Self
            current instance of the DatasetsSizeComparison check.
        """

        def condition(check_result: dict) -> ConditionResult:
            if check_result['Train'] < check_result['Test']:
                diff = check_result['Test'] - check_result['Train']
                return ConditionResult(False, f'Train dataset is smaller than test dataset by {diff}')
            else:
                return ConditionResult(True)

        return self.add_condition(
            name='Train dataset is not smaller than test dataset',
            condition_func=condition
        )
