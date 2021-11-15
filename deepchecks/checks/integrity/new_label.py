"""The data_sample_leakage_report check module."""

from deepchecks import Dataset
from deepchecks.base.check import CheckResult, TrainTestBaseCheck
from deepchecks.string_utils import format_percent

import pandas as pd

pd.options.mode.chained_assignment = None

__all__ = ['NewLabelTrainTest']


class NewLabelTrainTest(TrainTestBaseCheck):
    """Find new labels in test."""

    def run(self, train_dataset: Dataset, test_dataset: Dataset, model=None) -> CheckResult:
        """Run check.

        Args:
            train_dataset (Dataset): The training dataset object.
            test_dataset (Dataset): The test dataset object.
            model: any = None - not used in the check
        Returns:
            CheckResult: value is a dictionary that shows label column with new labels
                         displays a dataframe that label columns with new labels
        Raises:
            DeepchecksValueError: If the datasets are not a Dataset instance or do not contain label column
        """
        return self._new_label_train_test(train_dataset=train_dataset,
                                          test_dataset=test_dataset)

    def _new_label_train_test(self, train_dataset: Dataset, test_dataset: Dataset):
        test_dataset = Dataset.validate_dataset_or_dataframe(test_dataset)
        train_dataset = Dataset.validate_dataset_or_dataframe(train_dataset)
        test_dataset.validate_label(self.__class__.__name__)
        train_dataset.validate_label(self.__class__.__name__)
        test_dataset.validate_shared_label(train_dataset, self.__class__.__name__)

        label_column = train_dataset.validate_shared_label(test_dataset, self.__class__.__name__)

        n_test_samples = test_dataset.n_samples()

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

            result = {label_column: n_new_label / n_test_samples}
        else:
            display = None
            result = {}

        return CheckResult(result, check=self.run, display=display)
