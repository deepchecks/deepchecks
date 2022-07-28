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
"""The data_sample_leakage_report check module."""
from typing import Dict

import pandas as pd

from deepchecks.core import CheckResult, ConditionCategory, ConditionResult
from deepchecks.core.checks import ReduceMixin
from deepchecks.tabular import Context, TrainTestCheck
from deepchecks.utils.strings import format_percent

pd.options.mode.chained_assignment = None


__all__ = ['NewLabelTrainTest']


class NewLabelTrainTest(TrainTestCheck, ReduceMixin):
    """Find new labels in test."""

    def run_logic(self, context: Context) -> CheckResult:
        """Run check.

        Returns
        -------
        CheckResult
            value is a dictionary that shows label column with new labels
            displays a dataframe that label columns with new labels

        Raises
        ------
        DeepchecksValueError
            If the datasets are not a Dataset instance or do not contain label column
        """
        test_dataset = context.test
        train_dataset = context.train
        context.assert_classification_task()

        n_test_samples = test_dataset.n_samples

        train_label = train_dataset.label_col
        test_label = test_dataset.label_col

        unique_training_values = set(train_label.unique())
        unique_test_values = set(test_label.unique())

        new_labels = unique_test_values.difference(unique_training_values)

        if new_labels:
            new_labels = test_label[test_label.isin(new_labels)]
            n_new_label = len(new_labels)
            samples_per_label = dict(new_labels.value_counts())

            result = {
                'n_samples': n_test_samples,
                'n_new_labels_samples': n_new_label,
                'new_labels': sorted(samples_per_label.keys()),
                'n_samples_per_new_label': samples_per_label
            }

            if context.with_display:
                dataframe = pd.DataFrame(data=[[train_dataset.label_name, format_percent(n_new_label / n_test_samples),
                                                sorted(new_labels)]],
                                         columns=['Label column', 'Percent new labels in sample', 'New labels'])
                dataframe = dataframe.set_index(['Label column'])
                display = dataframe
            else:
                display = None

        else:
            display = None
            result = {}

        return CheckResult(result, display=display)

    def reduce_output(self, check_result: CheckResult) -> Dict[str, float]:
        """Reduce check result value.

        Returns
        -------
        Dict[str, float]
            number of samples per each new label
        """
        return check_result.value['n_samples_per_new_label']

    def add_condition_new_labels_number_less_or_equal(self, max_new: int = 0):
        """Add condition - require label column's number of different new labels to be less or equal to the threshold.

        Parameters
        ----------
        max_new : int , default: 0
            Number of different new labels value types which is the maximum allowed.
        """
        def condition(result: Dict) -> ConditionResult:
            if result:
                new_labels = result['new_labels']
                num_new_labels = len(new_labels)
                details = f'Found {num_new_labels} new labels in test data: {new_labels}'
                category = ConditionCategory.PASS if num_new_labels <= max_new else ConditionCategory.FAIL
                return ConditionResult(category, details)
            return ConditionResult(ConditionCategory.PASS, 'No new labels found')

        return self.add_condition(f'Number of new label values is less or equal to {max_new}',
                                  condition)

    def add_condition_new_label_ratio_less_or_equal(self, max_ratio: float = 0):
        """Add condition - require label column's ratio of new label samples to be less or equal to the threshold.

        Parameters
        ----------
        max_ratio : float , default: 0
            Ratio of new label samples to total samples which is the maximum allowed.
        """
        def new_category_count_condition(result: Dict) -> ConditionResult:
            if result:
                new_labels = result['new_labels']
                new_label_ratio = result['n_new_labels_samples'] / result['n_samples']
                details = f'Found {format_percent(new_label_ratio)} of labels in test data are new labels: {new_labels}'
                category = ConditionCategory.PASS if new_label_ratio <= max_ratio else ConditionCategory.FAIL
                return ConditionResult(category, details)
            return ConditionResult(ConditionCategory.PASS, 'No new labels found')

        return self.add_condition(
            f'Ratio of samples with new label is less or equal to {format_percent(max_ratio)}',
            new_category_count_condition)
