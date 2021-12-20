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
from typing import Union, List, Dict
import pandas as pd

from deepchecks import Dataset
from deepchecks.base.check import CheckResult, TrainTestBaseCheck, ConditionResult
from deepchecks.utils.strings import format_percent, format_columns_for_condition
from deepchecks.utils.typing import Hashable


__all__ = ['CategoryMismatchTrainTest']


class CategoryMismatchTrainTest(TrainTestBaseCheck):
    """Find new categories in the test set.

    Args:
        columns (Union[Hashable, List[Hashable]]):
            Columns to check, if none are given checks all columns except ignored ones.
        ignore_columns (Union[Hashable, List[Hashable]]):
            Columns to ignore, if none given checks based on columns
            variable.
    """

    def __init__(
        self,
        columns: Union[Hashable, List[Hashable], None] = None,
        ignore_columns: Union[Hashable, List[Hashable], None] = None
    ):
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

        features = test_dataset.validate_shared_features(train_dataset)
        cat_features = train_dataset.validate_shared_categorical_features(test_dataset)

        test_dataset = test_dataset.filter_columns_with_validation(self.columns, self.ignore_columns)
        train_dataset = train_dataset.filter_columns_with_validation(self.columns, self.ignore_columns)

        if set(features).symmetric_difference(set(test_dataset.features)):
            cat_features = test_dataset.features

        new_categories = []
        n_test_samples = test_dataset.n_samples

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
                                       n_new_cat,
                                       n_test_samples,
                                       sorted(new_category_values)])

        if new_categories:
            dataframe = pd.DataFrame(data=[[new_category[0],
                                            format_percent(new_category[1]/new_category[2]),
                                            new_category[3]]
                                           for new_category in new_categories],
                                     columns=['Column', 'Percent of new categories in sample', 'New categories'])
            dataframe = dataframe.set_index(['Column'])

            display = dataframe

            new_categories = dict(map(lambda category: (category[0], {
                'n_new': category[1],
                'n_total_samples': category[2],
                'new_categories': category[3]
            }), new_categories))
        else:
            display = None
            new_categories = {}
        return CheckResult(new_categories, display=display)

    def add_condition_new_categories_not_greater_than(self, max_new: int = 0):
        """Add condition - require column not to have greater than given number of different new categories.

        Args:
            max_new (int): Number of different categories value types which is the maximum allowed.
        """
        def condition(result: Dict) -> ConditionResult:
            not_passing_columns = []
            for column_name in result.keys():
                column = result[column_name]
                num_categories = len(column['new_categories'])
                if num_categories > max_new:
                    not_passing_columns.append(column_name)
            if not_passing_columns:
                not_passing_str = ', '.join(map(str, not_passing_columns))
                return ConditionResult(False,
                                       f'Found columns with more than {max_new} new categories: '
                                       f'{not_passing_str}')
            else:
                return ConditionResult(True)

        column_names = format_columns_for_condition(self.columns, self.ignore_columns)
        return self.add_condition(f'Number of new category values is not greater than {max_new} for {column_names}',
                                  condition)

    def add_condition_new_category_ratio_not_greater_than(self, max_ratio: float = 0):
        """Add condition - require column not to have greater than given number of different new categories.

        Args:
            max_ratio (int): Number of different categories value types which is the maximum allowed.
        """
        def new_category_count_condition(result: Dict) -> ConditionResult:
            not_passing_columns = []
            for column_name in result.keys():
                column = result[column_name]
                n_new_samples = column['n_new'] / column['n_total_samples']
                if n_new_samples > max_ratio:
                    not_passing_columns.append(column_name)
            if not_passing_columns:
                not_passing_str = ', '.join(map(str, not_passing_columns))
                return ConditionResult(False,
                                       f'Found columns with more than {format_percent(max_ratio)} new category samples:'
                                       f' {not_passing_str}')
            else:
                return ConditionResult(True)

        column_names = format_columns_for_condition(self.columns, self.ignore_columns)
        return self.add_condition(
            f'Ratio of samples with a new category is not greater than {format_percent(max_ratio)} for {column_names}',
            new_category_count_condition)
