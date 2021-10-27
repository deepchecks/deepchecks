"""The data_sample_leakage_report check module."""
from typing import Union, Iterable

from mlchecks import Dataset
from mlchecks.base.check import CheckResult, TrainValidationBaseCheck
from mlchecks.string_utils import format_percent

import pandas as pd


__all__ = ['new_category_train_validation', 'CategoryMismatchTrainValidation']


def new_category_train_validation(train_dataset: Dataset, validation_dataset: Dataset,
                                  columns: Union[str, Iterable[str]] = None,
                                  ignore_columns: Union[str, Iterable[str]] = None):
    """Find new categories in validation.

    Args:
        train_dataset (Dataset): The training dataset object.
        validation_dataset (Dataset): The validation dataset object.
        columns (Union[str, Iterable[str]]): Columns to check, if none are given checks all columns except ignored ones.
        ignore_columns (Union[str, Iterable[str]]): Columns to ignore, if none given checks based on columns variable

    Returns:
        CheckResult: value is a dictionary that shows columns with new categories
                     displays a dataframe that shows columns with new categories

    Raises:
        MLChecksValueError: If the object is not a Dataset instance

    """
    self = new_category_train_validation

    validation_dataset = Dataset.validate_dataset_or_dataframe(validation_dataset)
    train_dataset = Dataset.validate_dataset_or_dataframe(train_dataset)

    features = validation_dataset.validate_shared_features(train_dataset, self.__name__)

    cat_features = train_dataset.validate_shared_categorical_features(validation_dataset, self.__name__)

    validation_dataset = validation_dataset.filter_columns_with_validation(columns, ignore_columns)
    train_dataset = train_dataset.filter_columns_with_validation(columns, ignore_columns)

    if set(features).symmetric_difference(set(validation_dataset.features())):
        cat_features = validation_dataset.features()

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
                                   n_new_cat / n_validation_samples,
                                   sorted(new_category_values)])

    if new_categories:
        dataframe = pd.DataFrame(data=[[new_category[0], format_percent(new_category[1]), new_category[2]]
                                       for new_category in new_categories],
                                 columns=['Column', 'Percent of new categories in sample', 'New categories'])
        dataframe = dataframe.set_index(['Column'])

        display = dataframe

        new_categories = dict(map(lambda category: (category[0], category[1]), new_categories))
    else:
        display = None
        new_categories = {}
    return CheckResult(new_categories, check=self, display=display)


class CategoryMismatchTrainValidation(TrainValidationBaseCheck):
    """Find new categories in validation.

    Args:
        columns (Union[str, Iterable[str]]): Columns to check, if none are given checks all columns except ignored ones.
        ignore_columns (Union[str, Iterable[str]]): Columns to ignore, if none given checks based on columns variable

    """

    def run(self, train_dataset: Dataset, validation_dataset: Dataset, model=None) -> CheckResult:
        """Find new categories in validation.

        Args:
            train_dataset (Dataset): The training dataset object.
            validation_dataset (Dataset): The validation dataset object.
            model: any = None - not used in the check
        Returns:
            CheckResult: value is a dictionary that shows columns with new categories
                         displays a dataframe that shows columns with new categories
        """
        return new_category_train_validation(train_dataset=train_dataset,
                                             validation_dataset=validation_dataset,
                                             **self.params)
