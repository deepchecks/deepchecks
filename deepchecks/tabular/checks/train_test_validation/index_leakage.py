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
"""The index_leakage check module."""
import pandas as pd

from deepchecks.core import CheckResult, ConditionResult
from deepchecks.core.condition import ConditionCategory
from deepchecks.tabular import Context, TrainTestCheck
from deepchecks.utils.strings import format_percent

__all__ = ['IndexTrainTestLeakage']


class IndexTrainTestLeakage(TrainTestCheck):
    """Check if test indexes are present in train data.

    Parameters
    ----------
    n_to_show : int , default: 5
        Number of samples with same indices in train and test to show.
    """

    def __init__(self, n_to_show: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.n_index_to_show = n_to_show

    def run_logic(self, context: Context) -> CheckResult:
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

        train_dataset.assert_index()
        train_index = train_dataset.index_col
        val_index = test_dataset.index_col

        index_intersection = list(set(train_index).intersection(val_index))
        if len(index_intersection) > 0:
            size_in_test = len(index_intersection) / test_dataset.n_samples
            if context.with_display:
                text = f'{size_in_test:.1%} of test data indexes appear in training data'
                table = pd.DataFrame([[list(index_intersection)[:self.n_index_to_show]]],
                                     index=['Sample of test indexes in train:'])
                display = [text, table]
            else:
                display = None
        else:
            size_in_test = 0
            display = None

        return CheckResult(value=size_in_test, header='Index Train-Test Leakage', display=display)

    def add_condition_ratio_less_or_equal(self, max_ratio: float = 0):
        """Add condition - require index leakage ratio to be less or equal to threshold.

        Parameters
        ----------
        max_ratio : float , default: 0
            Maximum ratio of index leakage.
        """
        def max_ratio_condition(result: float) -> ConditionResult:
            details = f'Found {format_percent(result)} of index leakage' if result > 0 else 'No index leakage found'
            category = ConditionCategory.PASS if result <= max_ratio else ConditionCategory.FAIL
            return ConditionResult(category, details)

        return self.add_condition(f'Ratio of leaking indices is less or equal to {format_percent(max_ratio)}',
                                  max_ratio_condition)
