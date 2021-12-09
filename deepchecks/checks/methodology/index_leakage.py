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
"""The index_leakage check module."""
import pandas as pd

from deepchecks import CheckResult, Dataset, TrainTestBaseCheck
from deepchecks.base.check import ConditionResult
from deepchecks.utils.strings import format_percent


__all__ = ['IndexTrainTestLeakage']


class IndexTrainTestLeakage(TrainTestBaseCheck):
    """Check if test indexes are present in train data.

    Args:
        n_index_to_show (int): Number of common indexes to show.
    """

    def __init__(self, n_index_to_show: int = 5):
        super().__init__()
        self.n_index_to_show = n_index_to_show

    def run(self, train_dataset: Dataset, test_dataset: Dataset, model=None) -> CheckResult:
        """Run check.

        Arguments:
            train_dataset (Dataset): The training dataset object. Must contain an index.
            test_dataset (Dataset): The test dataset object. Must contain an index.
            model: any = None - not used in the check

        Returns:
           CheckResult:
                - value is the ratio of index leakage.
                - data is html display of the checks' textual result.

        Raises:
            DeepchecksValueError: If the if one of the datasets is not a Dataset instance with an index
        """
        return self._index_train_test_leakage(train_dataset, test_dataset)

    def _index_train_test_leakage(self, train_dataset: Dataset, test_dataset: Dataset):
        train_dataset = Dataset.validate_dataset(train_dataset)
        test_dataset = Dataset.validate_dataset(test_dataset)
        train_dataset.validate_index()
        test_dataset.validate_index()

        train_index = train_dataset.index_col
        val_index = test_dataset.index_col

        index_intersection = list(set(train_index).intersection(val_index))
        if len(index_intersection) > 0:
            size_in_test = len(index_intersection) / test_dataset.n_samples
            text = f'{size_in_test:.1%} of test data indexes appear in training data'
            table = pd.DataFrame([[list(index_intersection)[:self.n_index_to_show]]],
                                 index=['Sample of test indexes in train:'])
            display = [text, table]
        else:
            size_in_test = 0
            display = None

        return CheckResult(value=size_in_test, header='Index Train-Test Leakage', display=display)

    def add_condition_ratio_not_greater_than(self, max_ratio: float = 0):
        """Add condition - require index leakage ratio to not surpass max_ratio.

        Args:
            max_ratio (float): Maximum ratio of index leakage.
        """
        def max_ratio_condition(result: float) -> ConditionResult:
            if result > max_ratio:
                return ConditionResult(False, f'Found {format_percent(result)} of index leakage')
            else:
                return ConditionResult(True)

        return self.add_condition(f'Index leakage is not greater than {format_percent(max_ratio)}',
                                  max_ratio_condition)
