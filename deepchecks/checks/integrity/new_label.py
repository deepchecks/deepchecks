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
from typing import Dict, Any
import pandas as pd

from deepchecks import Dataset
from deepchecks.base.check import CheckResult, TrainTestBaseCheck, ConditionResult
from deepchecks.errors import ModelValidationError
from deepchecks.utils.metrics import ModelType
from deepchecks.utils.strings import format_percent

pd.options.mode.chained_assignment = None


__all__ = ['NewLabelTrainTest']


class NewLabelTrainTest(TrainTestBaseCheck):
    """Find new labels in test."""

    def run(self, train_dataset: Dataset, test_dataset: Dataset, model=None) -> CheckResult:
        """Run check.

        Args:
            train_dataset (Dataset): The training dataset object.
            test_dataset (Dataset): The test dataset object.
            model (any): used to check task type (default: None)

        Returns:
            CheckResult: value is a dictionary that shows label column with new labels
            displays a dataframe that label columns with new labels

        Raises:
            DeepchecksValueError: If the datasets are not a Dataset instance or do not contain label column
        """
        return self._new_label_train_test(train_dataset=train_dataset,
                                          test_dataset=test_dataset,
                                          model=model)

    def _new_label_train_test(self, train_dataset: Dataset, test_dataset: Dataset, model: Any = None):
        test_dataset = Dataset.ensure_not_empty_dataset(test_dataset)
        train_dataset = Dataset.ensure_not_empty_dataset(train_dataset)

        test_label = self._dataset_has_label(test_dataset)
        train_label = self._dataset_has_label(train_dataset)
        label_column = self._datasets_share_label([test_dataset, train_dataset])

        if model:
            self._verify_model_type(model, train_dataset, [ModelType.MULTICLASS, ModelType.BINARY])
        elif train_dataset.label_type == 'regression_label':
            raise ModelValidationError('Check is irrelevant for for the regression tasks')

        n_test_samples = test_dataset.n_samples

        train_label = train_dataset.data[label_column]
        test_label = test_dataset.data[label_column]

        unique_training_values = set(train_label.unique())
        unique_test_values = set(test_label.unique())

        new_labels = unique_test_values.difference(unique_training_values)

        if new_labels:
            n_new_label = len(test_label[test_label.isin(new_labels)])

            dataframe = pd.DataFrame(data=[[label_column, format_percent(n_new_label / n_test_samples),
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

        Args:
            max_new (int): Number of different new labels value types which is the maximum allowed.
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

        Args:
            max_ratio (int): Ratio of new label samples to total samples which is the maximum allowed.
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
