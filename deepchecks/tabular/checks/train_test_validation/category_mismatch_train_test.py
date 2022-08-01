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
from deepchecks.core.checks import ReduceMixin
from deepchecks.tabular import Context, TrainTestCheck
from deepchecks.tabular.utils.messages import get_condition_passed_message
from deepchecks.utils.dataframes import select_from_dataframe
from deepchecks.utils.strings import format_percent
from deepchecks.utils.typing import Hashable

__all__ = ['CategoryMismatchTrainTest']


class CategoryMismatchTrainTest(TrainTestCheck, ReduceMixin):
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
        self.max_features_to_show = max_features_to_show  # TODO: attr is not used, remove it
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
                new_categories_ratio = sum(new_category_counts.values()) / n_test_samples
                sorted_new_categories = dict(sorted(new_category_counts.items(), key=lambda x: x[1], reverse=True))
                new_categories[feature] = sorted_new_categories
                if context.with_display:
                    display_data.append([feature, len(new_category_values), new_categories_ratio,
                                        list(sorted_new_categories.keys())[:self.max_new_categories_to_show]])
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
            display['Percent of new categories in sample'] = display['Percent of new categories in sample'].apply(
                format_percent)

        else:
            display = None
        return CheckResult({'new_categories': new_categories, 'test_count': n_test_samples}, display=display)

    def reduce_output(self, check_result: CheckResult) -> Dict[str, float]:
        """Reduce check result value.

        Returns
        -------
        Dict[str, float]
            number of new categories per feature
        """
        return {
            feature_name: sum(categories.values())
            for feature_name, categories in check_result.value['new_categories'].items()
        }

    def add_condition_new_categories_less_or_equal(self, max_new: int = 0):
        """Add condition - require column's number of different new categories to be less or equal to threshold.

        Parameters
        ----------
        max_new : int , default: 0
            Number of different categories value types which is the maximum allowed.
        """
        def condition(result: Dict) -> ConditionResult:
            columns_new_categories = result['new_categories']
            num_new_per_column = [(feature, len(new_categories)) for feature, new_categories
                                  in columns_new_categories.items() if len(new_categories) > 0]
            sorted_columns = sorted(num_new_per_column, key=lambda x: x[1], reverse=True)
            failing = [(feature, num_new) for feature, num_new in sorted_columns if num_new > max_new]
            if failing:
                return ConditionResult(ConditionCategory.FAIL,
                                       f'Found {len(failing)} out of {len(columns_new_categories)} columns with number'
                                       f' of new categories above threshold:\n{dict(failing)}')
            else:
                details = get_condition_passed_message(result)
                if len(sorted_columns) > 0:
                    details += f'. Top columns with new categories count:\n{dict(sorted_columns[:5])}'
                return ConditionResult(ConditionCategory.PASS, details)

        return self.add_condition(f'Number of new category values is less or equal to {max_new}',
                                  condition)

    def add_condition_new_category_ratio_less_or_equal(self, max_ratio: float = 0):
        """Add condition - require column's ratio of instances with new categories to be less or equal to threshold.

        Parameters
        ----------
        max_ratio : float , default: 0
            Number of different categories value types which is the maximum allowed.
        """
        def new_category_count_condition(result: Dict) -> ConditionResult:
            columns_new_categories = result['new_categories']
            ratio_new_per_column = [(feature, sum(new_categories.values()) / result['test_count'])
                                    for feature, new_categories in columns_new_categories.items()]
            sorted_columns = sorted(ratio_new_per_column, key=lambda x: x[1], reverse=True)
            failing = [(feature, format_percent(ratio_new)) for feature, ratio_new in sorted_columns
                       if ratio_new > max_ratio]

            if failing:
                return ConditionResult(ConditionCategory.FAIL,
                                       f'Found {len(failing)} out of {len(columns_new_categories)} columns with ratio '
                                       f'of new category samples above threshold:\n{dict(failing)}')
            else:
                details = get_condition_passed_message(result)
                if len(sorted_columns) > 0:
                    columns_to_show = {feature: format_percent(ratio) for feature, ratio in sorted_columns[:5]}
                    details += f'. Top columns with new categories ratio:\n{columns_to_show}'
                return ConditionResult(ConditionCategory.PASS, details)

        return self.add_condition(
            f'Ratio of samples with a new category is less or equal to {format_percent(max_ratio)}',
            new_category_count_condition)
