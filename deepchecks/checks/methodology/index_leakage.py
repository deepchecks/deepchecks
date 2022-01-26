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

from deepchecks.base.check_context import CheckRunContext
from deepchecks import CheckResult, TrainTestBaseCheck
from deepchecks.base.check import ConditionResult
from deepchecks.utils.strings import format_percent


__all__ = ['IndexTrainTestLeakage']


class IndexTrainTestLeakage(TrainTestBaseCheck):
    """Check if test indexes are present in train data.

    Parameters
    ----------
    n_index_to_show : int , default: 5
        Number of common indexes to show.
    """

    def __init__(self, n_index_to_show: int = 5):
        super().__init__()
        self.n_index_to_show = n_index_to_show

    def run_logic(self, context: CheckRunContext) -> CheckResult:
        """Run check.

        Returns
        -------
        CheckResult
            value is the ratio of index leakage.
            data is html display of the checks' textual result.

        Raises
        ------
        DeepchecksValueError
            If one of the datasets is not a Dataset instance with an index
        """
        train_dataset = context.train
        test_dataset = context.test

        context.assert_index_exists()
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

        Parameters
        ----------
        max_ratio : float , default: 0
            Maximum ratio of index leakage.
        """
        def max_ratio_condition(result: float) -> ConditionResult:
            if result > max_ratio:
                return ConditionResult(False, f'Found {format_percent(result)} of index leakage')
            else:
                return ConditionResult(True)

        return self.add_condition(f'Ratio of leaking indices is not greater than {format_percent(max_ratio)}',
                                  max_ratio_condition)
