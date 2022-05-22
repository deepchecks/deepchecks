# ----------------------------------------------------------------------------
# Copyright (C) 2021-2022 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""The data_sample_leakage_report check module."""
from typing import Dict, List, Union

import pandas as pd

from deepchecks.core import CheckResult, ConditionCategory, ConditionResult
from deepchecks.tabular import Context, TrainTestCheck
from deepchecks.utils.dataframes import select_from_dataframe
from deepchecks.utils.strings import format_percent
from deepchecks.utils.typing import Hashable

__all__ = ['CategoryMismatchTrainTest']


class CategoryMismatchTrainTest(TrainTestCheck):
    """Find new categories in the test set.

    Parameters
    ----------
    columns : Union[Hashable, List[Hashable]] , default: None
        Columns to check, if none are given checks all columns except ignored ones.
    ignore_columns : Union[Hashable, List[Hashable]] , default: None
        Columns to ignore, if none given checks based on columns variable.
    max_features_to_show : int , default: 5
        maximum features with new categories to show
    max_new_categories_to_show : int , default: 5
        maximum new categories to show in feature
    """

    def __init__(
        self,
        columns: Union[Hashable, List[Hashable], None] = None,
        ignore_columns: Union[Hashable, List[Hashable], None] = None,
        max_features_to_show: int = 5,
        max_new_categories_to_show: int = 5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.columns = columns
        self.ignore_columns = ignore_columns
        self.max_features_to_show = max_features_to_show
        self.max_new_categories_to_show = max_new_categories_to_show

    def run_logic(self, context: Context) -> CheckResult:
        """Run check.

        Returns
        -------
        CheckResult
            value is a dictionary that lists new categories for each cat feature with its count
            displays a dataframe that shows columns with new categories
        """
        test_dataset = context.test
        train_dataset = context.train
        cat_features = train_dataset.cat_features

        test_df = select_from_dataframe(test_dataset.data, self.columns, self.ignore_columns)
        train_df = select_from_dataframe(train_dataset.data, self.columns, self.ignore_columns)

        # After filtering the columns drop cat features that don't exist anymore
        cat_features = set(cat_features).intersection(set(train_df.columns))

        new_categories = {}
        display_data = []
        n_test_samples = test_dataset.n_samples

        for feature in cat_features:
            train_column = train_df[feature]
            test_column = test_df[feature]

            # np.nan != np.nan, so we remove these values if they exist in training
            if train_column.isna().any():
                test_column = test_column.dropna()

            unique_training_values = train_column.unique()
            unique_test_values = test_column.unique()

            new_category_values = sorted(list((set(unique_test_values) - set(unique_training_values))))
            if new_category_values:
                new_category_counts = dict(test_column.value_counts()[new_category_values])
                new_categories_ratio = sum(count for _, count in new_category_counts) / n_test_samples
                sorted_new_categories = dict(sorted(new_category_counts.items(), key=lambda x: x[1], reverse=True))
                new_categories[feature] = sorted_new_categories
                display_data.append([feature, len(new_category_values), new_categories_ratio,
                                     sorted_new_categories.keys()[:self.max_new_categories_to_show]])
            else:
                new_categories[feature] = {}

        # Display
        if display_data:
            display = pd.DataFrame(data=display_data,
                                   columns=['Column',
                                            'Number of new categories',
                                            'Percent of new categories in sample',
                                            'New categories examples'])\
                                    .set_index(['Column'])

        else:
            display = None
        return CheckResult(new_categories, display=display)

    def add_condition_new_categories_not_greater_than(self, max_new: int = 0):
        """Add condition - require column not to have greater than given number of different new categories.

        Parameters
        ----------
        max_new : int , default: 0
            Number of different categories value types which is the maximum allowed.
        """
        def condition(result: Dict) -> ConditionResult:
            # not_passing_columns = {feature: len(new_categories) for feature, new_categories in result.items()
            #                        if len(new_categories) > max_new}
            num_new_per_column = [(feature, len(new_categories)) for feature, new_categories in result.items()]
            sorted_columns = sorted(num_new_per_column, key=lambda x: x[1], reverse=True)
            failing = [(feature, num_new) for feature, num_new in sorted_columns if num_new > max_new]
            if failing:
                return ConditionResult(ConditionCategory.FAIL,
                                       f'Found {len(failing)} columns with number of new categories above threshold '
                                       f'out of {len(result)} categorical columns:\n{dict(failing)}')
            else:
                return ConditionResult(ConditionCategory.PASS, )

        return self.add_condition(f'Number of new category values is not greater than {max_new}',
                                  condition)

    def add_condition_new_category_ratio_not_greater_than(self, max_ratio: float = 0):
        """Add condition - require column not to have greater than given ratio of instances with new categories.

        Parameters
        ----------
        max_ratio : float , default: 0
            Number of different categories value types which is the maximum allowed.
        """
        def new_category_count_condition(result: Dict) -> ConditionResult:
            not_passing_columns = {}
            for column_name in result.keys():
                column = result[column_name]
                n_new_samples = column['n_new'] / column['n_total_samples']
                if n_new_samples > max_ratio:
                    not_passing_columns[column_name] = format_percent(n_new_samples)
            if not_passing_columns:
                return ConditionResult(ConditionCategory.FAIL,
                                       f'Found columns with ratio of new category samples above threshold: '
                                       f'{not_passing_columns}')
            else:
                return ConditionResult(ConditionCategory.PASS)

        return self.add_condition(
            f'Ratio of samples with a new category is not greater than {format_percent(max_ratio)}',
            new_category_count_condition)
