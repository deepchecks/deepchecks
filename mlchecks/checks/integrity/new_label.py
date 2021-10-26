"""The data_sample_leakage_report check module."""

from mlchecks import Dataset
from mlchecks.base.check import CheckResult, TrainValidationBaseCheck

import pandas as pd
pd.options.mode.chained_assignment = None

__all__ = ['new_label_train_validation', 'NewLabelTrainValidation']


def new_label_train_validation(validation_dataset: Dataset, train_dataset: Dataset):
    """Find new categories in validation.

    Args:
        train_dataset (Dataset): The training dataset object.
        validation_dataset (Dataset): The validation dataset object.
    Returns:
        CheckResult:

    Raises:
        MLChecksValueError: If the object is not a Dataset instance

    """
    self = new_label_train_validation
    validation_dataset = Dataset.validate_dataset_or_dataframe(validation_dataset)
    train_dataset = Dataset.validate_dataset_or_dataframe(train_dataset)
    validation_dataset.validate_shared_label(train_dataset, self.__name__)

    label_columns = train_dataset.validate_shared_label(validation_dataset, self.__name__)

    result = []
    n_validation_samples = validation_dataset.n_samples()

    for column in label_columns:
        train_column = train_dataset.data[column]
        validation_column = validation_dataset.data[column]

        unique_training_values = set(train_column.unique())
        unique_validation_values = set(validation_column.unique())

        new_labels = unique_validation_values.difference(unique_training_values)

        if new_labels:
            n_new_label = len(validation_column[validation_column.isin(new_labels)])

            result.append([column,
                                   n_new_label/n_validation_samples,
                                   new_labels])

    if result:
        dataframe = pd.DataFrame(data=result,
                                 columns=['label column', 'ratio of new labels in sample', 'new labels'])
        dataframe = dataframe.set_index(['column'])

        display = dataframe

        result = dict(map(lambda category: (category[0], category[1]), result))
    else:
        display = None
        result = {}
    return CheckResult(result, check=self, display=display)


class NewLabelTrainValidation(TrainValidationBaseCheck):
    """Find new categories in validation."""

    def run(self, validation_dataset: Dataset, train_dataset: Dataset) -> CheckResult:
        """Find new categories in validation.

        Args:
            train_dataset (Dataset): The training dataset object.
            validation_dataset (Dataset): The validation dataset object.
        Returns:
            CheckResult: value is a dataframe that shows columns with new categories
                         displays a dataframe that shows columns with new categories
        """
        return new_label_train_validation(validation_dataset=validation_dataset,
                                          train_dataset=train_dataset,
                                          **self.params)
