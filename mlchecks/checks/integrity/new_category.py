"""The data_sample_leakage_report check module."""
from typing import Union, Iterable
from mlchecks import Dataset
from mlchecks.base.check import CheckResult, TrainValidationBaseCheck
from mlchecks.string_utils import format_percent

import pandas as pd

__all__ = ['CategoryMismatchTrainValidation']


class CategoryMismatchTrainValidation(TrainValidationBaseCheck):
    """Find new categories in validation."""

    def __init__(self, columns: Union[str, Iterable[str]] = None, ignore_columns: Union[str, Iterable[str]] = None):
        """
        Initialize the CategoryMismatchTrainValidation class.

        Args:
            columns (Union[str, Iterable[str]]): Columns to check, if none are given checks all columns except ignored
            ones.
            ignore_columns (Union[str, Iterable[str]]): Columns to ignore, if none given checks based on columns
            variable.
        """
        super().__init__()
        self.columns = columns
        self.ignore_columns = ignore_columns

    def run(self, train_dataset: Dataset, validation_dataset: Dataset, model=None) -> CheckResult:
        """Run check.

        Args:
            train_dataset (Dataset): The training dataset object.
            validation_dataset (Dataset): The validation dataset object.
            model: any = None - not used in the check
        Returns:
            CheckResult: value is a dictionary that shows columns with new categories
                         displays a dataframe that shows columns with new categories
        """
        return self._new_category_train_validation(train_dataset=train_dataset,
                                                   validation_dataset=validation_dataset)

    def _new_category_train_validation(self, train_dataset: Dataset, validation_dataset: Dataset):
        """Run check.

        Args:
            train_dataset (Dataset): The training dataset object.
            validation_dataset (Dataset): The validation dataset object.

        Returns:
            CheckResult: value is a dictionary that shows columns with new categories
                         displays a dataframe that shows columns with new categories

        Raises:
            MLChecksValueError: If the object is not a Dataset instance

        """
        validation_dataset = Dataset.validate_dataset_or_dataframe(validation_dataset)
        train_dataset = Dataset.validate_dataset_or_dataframe(train_dataset)

        features = validation_dataset.validate_shared_features(train_dataset, self.__class__.__name__)
        cat_features = train_dataset.validate_shared_categorical_features(validation_dataset, self.__class__.__name__)

        validation_dataset = validation_dataset.filter_columns_with_validation(self.columns, self.ignore_columns)
        train_dataset = train_dataset.filter_columns_with_validation(self.columns, self.ignore_columns)

        if set(features).symmetric_difference(set(validation_dataset.features())):
            cat_features = validation_dataset.features()

        new_categories = []
        n_validation_samples = validation_dataset.n_samples()

        for feature in cat_features:
            train_column = train_dataset.data[feature]
            validation_column = validation_dataset.data[feature]

            # np.nan doesn't compare, so we remove these values if they exist in in training
            if train_column.isna().any():
                validation_column = validation_column.dropna()

            unique_training_values = train_column.unique()
            unique_validation_values = validation_column.unique()

            new_category_values = set(unique_validation_values).difference(set(unique_training_values))

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
        return CheckResult(new_categories, check=self.__class__, display=display)
