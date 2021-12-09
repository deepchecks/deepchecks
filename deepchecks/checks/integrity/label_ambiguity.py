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
"""module contains Data Duplicates check."""
from typing import Union, List

import pandas as pd

from deepchecks import Dataset, ConditionResult
from deepchecks.base.check import CheckResult, SingleDatasetBaseCheck
from deepchecks.utils.strings import format_percent
from deepchecks.utils.typing import Hashable


__all__ = ['LabelAmbiguity']


class LabelAmbiguity(SingleDatasetBaseCheck):
    """Find samples with multiple labels.

    Args:
        columns (Hashable, List[Hashable]):
            List of columns to check, if none given checks
            all columns Except ignored ones.
        ignore_columns (Hashable, List[Hashable]):
            List of columns to ignore, if none given checks
            based on columns variable.
        n_to_show (int):
            number of most common ambiguous samples to show.
    """

    def __init__(
        self,
        columns: Union[Hashable, List[Hashable], None] = None,
        ignore_columns: Union[Hashable, List[Hashable], None] = None,
        n_to_show: int = 5
    ):
        super().__init__()
        self.columns = columns
        self.ignore_columns = ignore_columns
        self.n_to_show = n_to_show

    def run(self, dataset: Dataset, model=None) -> CheckResult:
        """Run check.

        Args:
          dataset(Dataset): any dataset.

        Returns:
          (CheckResult): percentage of ambiguous samples and display of the top n_to_show most ambiguous.
        """
        dataset: Dataset = Dataset.validate_dataset(dataset)
        dataset = dataset.filter_columns_with_validation(self.columns, self.ignore_columns)
        dataset.validate_label()

        label_col = dataset.label_name

        group_unique_data = dataset.data.groupby(dataset.features, dropna=False)
        group_unique_labels = group_unique_data.nunique()[label_col]

        num_ambiguous = 0
        display = pd.DataFrame(columns=[dataset.label_name, *dataset.features])

        for num_labels, group_data in sorted(zip(group_unique_labels, group_unique_data),
                                             key=lambda x: x[0], reverse=True):
            if num_labels == 1:
                break

            group_df = group_data[1]
            sample_values = dict(group_df[dataset.features].iloc[0])
            labels = list(group_df[label_col].unique())
            n_data_sample = group_df.shape[0]
            num_ambiguous += n_data_sample

            display = display.append({label_col: labels, **sample_values}, ignore_index=True)

        display.set_index(label_col)

        display = None if display.empty else display.head(self.n_to_show)

        percent_ambiguous = num_ambiguous/dataset.n_samples

        return CheckResult(value=percent_ambiguous, display=display)

    def add_condition_ambiguous_sample_ratio_not_greater_than(self, max_ratio=0):
        """Add condition - require samples with multiple labels to not be more than max_ratio.

        Args:
            max_ratio (float): Maximum ratio of samples with multiple labels.
        """
        def max_ratio_condition(result: float) -> ConditionResult:
            if result > max_ratio:
                return ConditionResult(False, f'Found {format_percent(result)} samples with multiple labels')
            else:
                return ConditionResult(True)

        return self.add_condition(f'Ambiguous sample ratio is not greater than {format_percent(max_ratio)}',
                                  max_ratio_condition)
