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
"""module contains Data Duplicates check."""
from typing import List, TypedDict, Union

import pandas as pd

from deepchecks.core import CheckResult, ConditionCategory, ConditionResult
from deepchecks.tabular import Context, SingleDatasetCheck
from deepchecks.utils.strings import format_percent
from deepchecks.utils.typing import Hashable

__all__ = ['ConflictingLabels']


class ResultValue(TypedDict):
    percent: float
    samples: List[pd.DataFrame]


class ConflictingLabels(SingleDatasetCheck):
    """Find samples which have the exact same features' values but different labels.

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
        n_to_show: int = 5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.columns = columns
        self.ignore_columns = ignore_columns
        self.n_to_show = n_to_show

    def run_logic(self, context: Context, dataset_kind) -> CheckResult:
        """Run check.

        Returns
        -------
        CheckResult
            percentage of ambiguous samples and display of the top n_to_show most ambiguous.
        """
        dataset = context.get_data_by_kind(dataset_kind)
        context.assert_classification_task()
        dataset.assert_label()
        features = dataset.features
        label_name = dataset.label_name
        with_display = context.with_display

        dataset = dataset.select(self.columns, self.ignore_columns, keep_label=True)

        # HACK: pandas have bug with groupby on category dtypes, so until it fixed, change dtypes manually
        df = dataset.data
        category_columns = df.dtypes[df.dtypes == 'category'].index.tolist()
        if category_columns:
            df = df.astype({c: 'object' for c in category_columns})

        group_unique_data = df.groupby(features, dropna=False)
        group_unique_labels = group_unique_data.nunique()[label_name]

        num_ambiguous = 0
        ambiguous_label_name = 'Observed Labels'
        samples = []
        display_samples = []

        data = sorted(
            zip(group_unique_labels, group_unique_data),
            key=lambda x: x[0],
            reverse=True
        )

        for num_labels, group_data in data:
            if num_labels == 1:
                continue

            group_df = group_data[1]

            n_data_sample = group_df.shape[0]
            num_ambiguous += n_data_sample
            samples.append(group_df.loc[:, [label_name, *features]].copy())

            if with_display is True:
                display_sample = dict(group_df[features].iloc[0])
                ambiguous_labels = tuple(sorted(group_df[label_name].unique()))
                display_sample[ambiguous_label_name] = ambiguous_labels
                display_samples.append(display_sample)

        if not (with_display is True and len(display_samples) > 0):
            display = None
        else:
            display = pd.DataFrame.from_records(display_samples[:self.n_to_show])
            display.set_index(ambiguous_label_name, inplace=True)
            display = [
                'Each row in the table shows an example of a data sample '
                'and the its observed labels as found in the dataset. '
                f'Showing top {self.n_to_show} of {display.shape[0]}',
                display
            ]

        return CheckResult(
            display=display,
            value=ResultValue(
                percent=num_ambiguous / dataset.n_samples,
                samples=samples,
            )
        )

    def add_condition_ratio_of_conflicting_labels_less_or_equal(self, max_ratio=0):
        """Add condition - require ratio of samples with conflicting labels less or equal to max_ratio.

        Parameters
        ----------
        max_ratio : float , default: 0
            Maximum ratio of samples with multiple labels.
        """
        def max_ratio_condition(result: ResultValue) -> ConditionResult:
            percent = result['percent']
            details = f'Ratio of samples with conflicting labels: {format_percent(percent)}'
            category = ConditionCategory.PASS if percent <= max_ratio else ConditionCategory.FAIL
            return ConditionResult(category, details)

        return self.add_condition(f'Ambiguous sample ratio is less or equal to {format_percent(max_ratio)}',
                                  max_ratio_condition)
