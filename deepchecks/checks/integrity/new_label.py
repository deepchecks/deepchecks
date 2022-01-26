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
"""The data_sample_leakage_report check module."""
from typing import Dict
import pandas as pd

from deepchecks.base.check_context import CheckRunContext
from deepchecks.base.check import CheckResult, TrainTestBaseCheck, ConditionResult
from deepchecks.utils.strings import format_percent

pd.options.mode.chained_assignment = None


__all__ = ['NewLabelTrainTest']


class NewLabelTrainTest(TrainTestBaseCheck):
    """Find new labels in test."""

    def run_logic(self, context: CheckRunContext) -> CheckResult:
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
        label_name = context.label_name

        context.assert_classification_task()

        n_test_samples = test_dataset.n_samples

        train_label = train_dataset.data[label_name]
        test_label = test_dataset.data[label_name]

        unique_training_values = set(train_label.unique())
        unique_test_values = set(test_label.unique())

        new_labels = unique_test_values.difference(unique_training_values)

        if new_labels:
            n_new_label = len(test_label[test_label.isin(new_labels)])

            dataframe = pd.DataFrame(data=[[label_name, format_percent(n_new_label / n_test_samples),
                                            sorted(new_labels)]],
                                     columns=['Label column', 'Percent new labels in sample', 'New labels'])
            dataframe = dataframe.set_index(['Label column'])

            display = dataframe

            result = {
                'n_samples': n_test_samples,
                'n_new_labels_samples': n_new_label,
                'new_labels': sorted(new_labels)
            }
        else:
            display = None
            result = {}

        return CheckResult(result, display=display)

    def add_condition_new_labels_not_greater_than(self, max_new: int = 0):
        """Add condition - require label column not to have greater than given number of different new labels.

        Parameters
        ----------
        max_new : int , default: 0
            Number of different new labels value types which is the maximum allowed.
        """
        def condition(result: Dict) -> ConditionResult:
            if result:
                new_labels = result['new_labels']
                num_new_labels = len(new_labels)
                if num_new_labels > max_new:
                    return ConditionResult(False,
                                           f'Found {num_new_labels} new labels: {new_labels}')
            return ConditionResult(True)

        return self.add_condition(f'Number of new label values is not greater than {max_new}',
                                  condition)

    def add_condition_new_label_ratio_not_greater_than(self, max_ratio: float = 0):
        """Add condition - require label column not to have greater than given number of ratio new label samples.

        Parameters
        ----------
        max_ratio : float , default: 0
            Ratio of new label samples to total samples which is the maximum allowed.
        """
        def new_category_count_condition(result: Dict) -> ConditionResult:
            if result:
                new_labels = result['new_labels']
                new_label_ratio = result['n_new_labels_samples'] / result['n_samples']
                if new_label_ratio > max_ratio:
                    return ConditionResult(False,
                                           f'Found new labels in test data: {new_labels}\n'
                                           f'making {format_percent(new_label_ratio)} of samples.')
            return ConditionResult(True)

        return self.add_condition(
            f'Ratio of samples with new label is not greater than {format_percent(max_ratio)}',
            new_category_count_condition)
