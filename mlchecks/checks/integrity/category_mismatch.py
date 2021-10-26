"""The data_sample_leakage_report check module."""

from mlchecks import Dataset
from mlchecks.base.check import CheckResult, TrainValidationBaseCheck

import pandas as pd
pd.options.mode.chained_assignment = None

__all__ = ['category_mismatch_train_validation', 'CategoryMismatchTrainValidation']


def category_mismatch_train_validation(validation_dataset: Dataset, train_dataset: Dataset):
    """Find new and missing categories in validation.

    Args:
        train_dataset (Dataset): The training dataset object.
        validation_dataset (Dataset): The validation dataset object.
    Returns:
        CheckResult:

    Raises:
        MLChecksValueError: If the object is not a Dataset instance

    """
    self = category_mismatch_train_validation
    validation_dataset = Dataset.validate_dataset_or_dataframe(validation_dataset)
    train_dataset = Dataset.validate_dataset_or_dataframe(train_dataset)
    validation_dataset.validate_shared_features(train_dataset, self.__name__)

    cat_features = train_dataset.validate_shared_categorical_features(validation_dataset, self.__name__)

    result = []

    for feature in cat_features:
        unique_training_values = set(train_dataset.data[feature].unique())
        unique_validation_values = set(validation_dataset.data[feature].unique())

        new_category_values = unique_validation_values.difference(unique_training_values)
        missing_category_values = unique_training_values.difference(unique_validation_values)
        shared = unique_training_values.intersection(unique_validation_values)

        if new_category_values or missing_category_values:
            result.append([feature,
                           new_category_values if new_category_values else None,
                           missing_category_values if missing_category_values else None,
                           shared if shared else None])

    dataframe = pd.DataFrame(data=result,
                             columns=['column', 'new categories', 'missing categories', 'shared categories'])
    dataframe = dataframe.set_index(['column'])

    display = dataframe if len(dataframe) else None

    return CheckResult(dataframe, check=self, display=display)


class CategoryMismatchTrainValidation(TrainValidationBaseCheck):
    """Find mismatching categories in validation."""

    def run(self, validation_dataset: Dataset, train_dataset: Dataset) -> CheckResult:
        """Find new and missing categories in validation.

        Args:
            train_dataset (Dataset): The training dataset object.
            validation_dataset (Dataset): The validation dataset object.
        Returns:
            CheckResult: value is a dataframe that shows columns with mismatching categories
                         displays a dataframe that shows columns with mismatching categories
        """
        return category_mismatch_train_validation(validation_dataset=validation_dataset,
                                                  train_dataset=train_dataset,
                                                  **self.params)
