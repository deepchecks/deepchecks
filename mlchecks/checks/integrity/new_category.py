"""The data_sample_leakage_report check module."""

from mlchecks import Dataset
from mlchecks.base.check import CheckResult, TrainValidationBaseCheck

import pandas as pd
pd.options.mode.chained_assignment = None

__all__ = ['new_category_train_validation', 'CategoryMismatchTrainValidation']


def new_category_train_validation(validation_dataset: Dataset, train_dataset: Dataset):
    """Find new categories in validation.

    Args:
        train_dataset (Dataset): The training dataset object.
        validation_dataset (Dataset): The validation dataset object.
    Returns:
        CheckResult:

    Raises:
        MLChecksValueError: If the object is not a Dataset instance

    """
    self = new_category_train_validation
    validation_dataset = Dataset.validate_dataset_or_dataframe(validation_dataset)
    train_dataset = Dataset.validate_dataset_or_dataframe(train_dataset)
    validation_dataset.validate_shared_features(train_dataset, self.__name__)

    cat_features = train_dataset.validate_shared_categorical_features(validation_dataset, self.__name__)

    new_categories = []
    n_validation_samples = validation_dataset.n_samples()

    for feature in cat_features:
        train_column = train_dataset.data[feature]
        validation_column = validation_dataset.data[feature]

        unique_training_values = set(train_column.unique())
        unique_validation_values = set(validation_column.unique())

        new_category_values = unique_validation_values.difference(unique_training_values)

        if new_category_values:
            n_new_cat = len(validation_column[validation_column.isin(new_category_values)])

            new_categories.append([feature,
                                   n_new_cat/n_validation_samples,
                                   new_category_values])

    if new_categories:
        dataframe = pd.DataFrame(data=new_categories,
                                 columns=['column', 'ratio of new categories in sample', 'new categories'])
        dataframe = dataframe.set_index(['column'])

        display = dataframe

        new_categories = dict(map(lambda category: (category[0], category[1]), new_categories))
    else:
        display = None
        new_categories = {}
    return CheckResult(new_categories, check=self, display=display)


class CategoryMismatchTrainValidation(TrainValidationBaseCheck):
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
        return new_category_train_validation(validation_dataset=validation_dataset,
                                             train_dataset=train_dataset,
                                             **self.params)
