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

from deepchecks.base.check_context import CheckRunContext
from deepchecks import ConditionResult
from deepchecks.base.check import CheckResult, SingleDatasetBaseCheck
from deepchecks.utils.strings import format_percent
from deepchecks.utils.typing import Hashable

__all__ = ['LabelAmbiguity']


class LabelAmbiguity(SingleDatasetBaseCheck):
    """Find samples with multiple labels.

    Parameters
    ----------
    columns : Union[Hashable, List[Hashable]] , default: None
        List of columns to check, if none given checks
        all columns Except ignored ones.
    ignore_columns : Union[Hashable, List[Hashable]] , default: None
        List of columns to ignore, if none given checks
        based on columns variable.
    n_to_show : int , default: 5
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

    def run_logic(self, context: CheckRunContext, dataset_type: str = 'train') -> CheckResult:
        """Run check.

        Returns
        -------
        CheckResult
            percentage of ambiguous samples and display of the top n_to_show most ambiguous.
        """
        if dataset_type == 'train':
            dataset = context.train
        else:
            dataset = context.test

        context.assert_classification_task()

        dataset = dataset.select(self.columns, self.ignore_columns, keep_label=True)

        label_col = context.label_name

        # HACK: pandas have bug with groupby on category dtypes, so until it fixed, change dtypes manually
        df = dataset.data
        category_columns = df.dtypes[df.dtypes == 'category'].index.tolist()
        if category_columns:
            df = df.astype({c: 'object' for c in category_columns})

        group_unique_data = df.groupby(dataset.features, dropna=False)
        group_unique_labels = group_unique_data.nunique()[label_col]

        num_ambiguous = 0
        ambiguous_label_name = 'Observed Labels'
        display = pd.DataFrame(columns=[ambiguous_label_name, *dataset.features])

        for num_labels, group_data in sorted(zip(group_unique_labels, group_unique_data),
                                             key=lambda x: x[0], reverse=True):
            if num_labels == 1:
                break

            group_df = group_data[1]
            sample_values = dict(group_df[dataset.features].iloc[0])
            labels = tuple(sorted(group_df[label_col].unique()))
            n_data_sample = group_df.shape[0]
            num_ambiguous += n_data_sample

            display = display.append({ambiguous_label_name: labels, **sample_values}, ignore_index=True)

        display = display.set_index(ambiguous_label_name)

        explanation = ('Each row in the table shows an example of a data sample '
                       'and the its observed labels as found in the dataset. '
                       f'Showing top {self.n_to_show} of {display.shape[0]}')

        display = None if display.empty else [explanation, display.head(self.n_to_show)]

        percent_ambiguous = num_ambiguous / dataset.n_samples

        return CheckResult(value=percent_ambiguous, display=display)

    def add_condition_ambiguous_sample_ratio_not_greater_than(self, max_ratio=0):
        """Add condition - require samples with multiple labels to not be more than max_ratio.

        Parameters
        ----------
        max_ratio : float , default: 0
            Maximum ratio of samples with multiple labels.
        """

        def max_ratio_condition(result: float) -> ConditionResult:
            if result > max_ratio:
                return ConditionResult(False, f'Found ratio of samples with multiple labels above threshold: '
                                              f'{format_percent(result)}')
            else:
                return ConditionResult(True)

        return self.add_condition(f'Ambiguous sample ratio is not greater than {format_percent(max_ratio)}',
                                  max_ratio_condition)
