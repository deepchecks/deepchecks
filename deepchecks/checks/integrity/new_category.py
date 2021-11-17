"""The data_sample_leakage_report check module."""
from typing import Union, Iterable
from deepchecks import Dataset
from deepchecks.base.check import CheckResult, TrainTestBaseCheck
from deepchecks.string_utils import format_percent

import pandas as pd

__all__ = ['CategoryMismatchTrainTest']


class CategoryMismatchTrainTest(TrainTestBaseCheck):
    """Find new categories in test."""

    def __init__(self, columns: Union[str, Iterable[str]] = None, ignore_columns: Union[str, Iterable[str]] = None):
        """
        Initialize the CategoryMismatchTrainTest class.

        Args:
            columns (Union[str, Iterable[str]]): Columns to check, if none are given checks all columns except ignored
            ones.
            ignore_columns (Union[str, Iterable[str]]): Columns to ignore, if none given checks based on columns
            variable.
        """
        super().__init__()
        self.columns = columns
        self.ignore_columns = ignore_columns

    def run(self, train_dataset: Dataset, test_dataset: Dataset, model=None) -> CheckResult:
        """Run check.

        Args:
            train_dataset (Dataset): The training dataset object.
            test_dataset (Dataset): The test dataset object.
            model: any = None - not used in the check
        Returns:
            CheckResult: value is a dictionary that shows columns with new categories
            displays a dataframe that shows columns with new categories
        """
        return self._new_category_train_test(train_dataset=train_dataset,
                                                   test_dataset=test_dataset)

    def _new_category_train_test(self, train_dataset: Dataset, test_dataset: Dataset):
        """Run check.

        Args:
            train_dataset (Dataset): The training dataset object.
            test_dataset (Dataset): The test dataset object.

        Returns:
            CheckResult: value is a dictionary that shows columns with new categories
                         displays a dataframe that shows columns with new categories

        Raises:
            DeepchecksValueError: If the object is not a Dataset instance

        """
        test_dataset = Dataset.validate_dataset_or_dataframe(test_dataset)
        train_dataset = Dataset.validate_dataset_or_dataframe(train_dataset)

        features = test_dataset.validate_shared_features(train_dataset, self.__class__.__name__)
        cat_features = train_dataset.validate_shared_categorical_features(test_dataset, self.__class__.__name__)

        test_dataset = test_dataset.filter_columns_with_validation(self.columns, self.ignore_columns)
        train_dataset = train_dataset.filter_columns_with_validation(self.columns, self.ignore_columns)

        if set(features).symmetric_difference(set(test_dataset.features())):
            cat_features = test_dataset.features()

        new_categories = []
        n_test_samples = test_dataset.n_samples()

        for feature in cat_features:
            train_column = train_dataset.data[feature]
            test_column = test_dataset.data[feature]

            # np.nan != np.nan, so we remove these values if they exist in training
            if train_column.isna().any():
                test_column = test_column.dropna()

            unique_training_values = train_column.unique()
            unique_test_values = test_column.unique()

            new_category_values = set(unique_test_values).difference(set(unique_training_values))

            if new_category_values:
                n_new_cat = len(test_column[test_column.isin(new_category_values)])

                new_categories.append([feature,
                                       n_new_cat / n_test_samples,
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
