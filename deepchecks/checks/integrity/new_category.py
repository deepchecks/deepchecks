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
from deepchecks.utils.strings import format_percent
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
        max_features_to_show (int): maximum features with new categories to show
        max_new_categories_to_show (int): maximum new categories to show in feature
    """

    def __init__(
        self,
        columns: Union[Hashable, List[Hashable], None] = None,
        ignore_columns: Union[Hashable, List[Hashable], None] = None,
        max_features_to_show: int = 5,
        max_new_categories_to_show: int = 5
    ):
        super().__init__()
        self.columns = columns
        self.ignore_columns = ignore_columns
        self.max_features_to_show = max_features_to_show
        self.max_new_categories_to_show = max_new_categories_to_show

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
        test_dataset = Dataset.ensure_not_empty_dataset(test_dataset, cast=True)
        train_dataset = Dataset.ensure_not_empty_dataset(train_dataset, cast=True)

        features = self._datasets_share_features([train_dataset, test_dataset])
        cat_features = self._datasets_share_categorical_features([train_dataset, test_dataset])

        test_dataset = test_dataset.select(self.columns, self.ignore_columns)
        train_dataset = train_dataset.select(self.columns, self.ignore_columns)

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

            new_category_values = sorted(set(unique_test_values).difference(set(unique_training_values)))
            new_category_samples = dict(test_column.value_counts()[new_category_values])
            sorted_new_categories = sorted(new_category_values,
                                           key=lambda x, count=new_category_samples: count[x],
                                           reverse=True)

            if new_category_values:
                n_new_cat = len(test_column[test_column.isin(new_category_values)])

                new_categories.append({'name': feature,
                                       'n_new': n_new_cat,
                                       'n_total_samples': n_test_samples,
                                       'new_categories': sorted_new_categories})

        if new_categories:
            dataframe = pd.DataFrame(data=[[new_category['name'],
                                            len(new_category['new_categories']),
                                            format_percent(
                                                new_category['n_new']/new_category['n_total_samples']),
                                            new_category['new_categories'][:self.max_new_categories_to_show]]
                                           for new_category in new_categories[:self.max_features_to_show]],
                                     columns=['Column',
                                              'Number of new categories',
                                              'Percent of new categories in sample',
                                              'New categories examples'])
            dataframe = dataframe.set_index(['Column'])

            display = dataframe

            new_categories = dict(map(lambda category: (category['name'], {
                'n_new': category['n_new'],
                'n_total_samples': category['n_total_samples'],
                'new_categories': category['new_categories']
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
            not_passing_columns = {}
            for column_name in result.keys():
                column = result[column_name]
                num_categories = len(column['new_categories'])
                if num_categories > max_new:
                    not_passing_columns[column_name] = num_categories
            if not_passing_columns:
                return ConditionResult(False,
                                       f'Found columns with number of new categories above threshold: '
                                       f'{not_passing_columns}')
            else:
                return ConditionResult(True)

        return self.add_condition(f'Number of new category values is not greater than {max_new}',
                                  condition)

    def add_condition_new_category_ratio_not_greater_than(self, max_ratio: float = 0):
        """Add condition - require column not to have greater than given number of different new categories.

        Args:
            max_ratio (int): Number of different categories value types which is the maximum allowed.
        """
        def new_category_count_condition(result: Dict) -> ConditionResult:
            not_passing_columns = {}
            for column_name in result.keys():
                column = result[column_name]
                n_new_samples = column['n_new'] / column['n_total_samples']
                if n_new_samples > max_ratio:
                    not_passing_columns[column_name] = format_percent(n_new_samples)
            if not_passing_columns:
                return ConditionResult(False,
                                       f'Found columns with ratio of new category samples above threshold: '
                                       f'{not_passing_columns}')
            else:
                return ConditionResult(True)

        return self.add_condition(
            f'Ratio of samples with a new category is not greater than {format_percent(max_ratio)}',
            new_category_count_condition)
