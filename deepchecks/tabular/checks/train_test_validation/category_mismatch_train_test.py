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
from deepchecks.core.reduce_classes import ReduceFeatureMixin
from deepchecks.tabular import Context, TrainTestCheck
from deepchecks.tabular._shared_docs import docstrings
from deepchecks.tabular.utils.messages import get_condition_passed_message
from deepchecks.utils.dataframes import select_from_dataframe
from deepchecks.utils.strings import format_number, format_percent
from deepchecks.utils.typing import Hashable

__all__ = ['CategoryMismatchTrainTest']


@docstrings
class CategoryMismatchTrainTest(TrainTestCheck, ReduceFeatureMixin):
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
    aggregation_method: str, default: 'max'
        {feature_aggregation_method_argument:2*indent}
    n_samples : int , default: 10_000_000
        number of samples to use for this check.
    random_state : int, default: 42
        random seed for all check internals.
    """

    def __init__(
            self,
            columns: Union[Hashable, List[Hashable], None] = None,
            ignore_columns: Union[Hashable, List[Hashable], None] = None,
            max_features_to_show: int = 5,
            max_new_categories_to_show: int = 5,
            aggregation_method='max',
            n_samples: int = 10_000_000,
            random_state: int = 42,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.columns = columns
        self.ignore_columns = ignore_columns
        self.max_features_to_show = max_features_to_show
        self.max_new_categories_to_show = max_new_categories_to_show
        self.aggregation_method = aggregation_method
        self.n_samples = n_samples
        self.random_state = random_state

    def run_logic(self, context: Context) -> CheckResult:
        """Run check.

        Returns
        -------
        CheckResult
            value is a dataframe that lists new categories for each cat feature with its count
            displays a dataframe that shows columns with new categories
        """
        test_dataset = context.test.sample(self.n_samples, random_state=self.random_state)
        train_dataset = context.train.sample(self.n_samples, random_state=self.random_state)
        cat_features = train_dataset.cat_features

        test_df = select_from_dataframe(test_dataset.data, self.columns, self.ignore_columns)
        train_df = select_from_dataframe(train_dataset.data, self.columns, self.ignore_columns)

        # After filtering the columns drop cat features that don't exist anymore
        cat_features = set(cat_features).intersection(set(train_df.columns))
        feature_importance = pd.Series(index=list(cat_features), dtype=object) if context.feature_importance is None \
            else context.feature_importance

        result_data = []
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
                result_data.append([feature, len(new_category_values), new_categories_ratio,
                                    list(sorted_new_categories.keys()), feature_importance[feature]])
            else:
                result_data.append([feature, 0, 0, [], feature_importance[feature]])

        result_data = pd.DataFrame(data=result_data,
                                   columns=['Column',
                                            'Number of new categories',
                                            'Percent of new categories in sample',
                                            'New categories in column',
                                            'Feature importance']).set_index(['Column'])
        result_data.sort_values(by='Percent of new categories in sample')
        if all(feature_importance.isna()):
            result_data.drop('Feature importance', axis=1, inplace=True)

        if context.with_display:
            display = result_data.copy()
            display['Percent of new categories in sample'] = display['Percent of new categories in sample'].apply(
                format_percent)
            display['Number of new categories'] = display['Number of new categories'].apply(format_number)
            display['New categories examples'] = display['New categories in column']. \
                apply(lambda x: x[:self.max_new_categories_to_show])
            display.drop('New categories in column', axis=1, inplace=True)
            display = display.iloc[:self.max_features_to_show, :]
        else:
            display = None

        return CheckResult(result_data, display=display)

    def reduce_output(self, check_result: CheckResult) -> Dict[str, float]:
        """Return an aggregated drift score based on aggregation method defined."""
        feature_importance = check_result.value['Feature importance'] if 'Feature importance' \
                                                                         in check_result.value.columns else None
        values = check_result.value['Percent of new categories in sample']
        return self.feature_reduce(self.aggregation_method, values, feature_importance, 'New Categories Ratio')

    def add_condition_new_categories_less_or_equal(self, max_new: int = 0):
        """Add condition - require column's number of different new categories to be less or equal to threshold.

        Parameters
        ----------
        max_new : int , default: 0
            Number of different categories value types which is the maximum allowed.
        """

        def condition(result: pd.DataFrame) -> ConditionResult:
            failing = result[result['Number of new categories'] > max_new]['Number of new categories']
            if len(failing) > 0:
                return ConditionResult(ConditionCategory.FAIL,
                                       f'Found {len(failing)} columns with number'
                                       f' of new categories above threshold: \n{dict(failing)}')
            else:
                details = get_condition_passed_message(len(result))
                if any(result['Number of new categories'] > 0):
                    new_categories_columns = dict(
                        result[result['Number of new categories'] > 0]['Number of new categories'][:5])
                    details += f'. Top columns with new categories count: \n{new_categories_columns}'
                return ConditionResult(ConditionCategory.PASS, details)

        return self.add_condition(f'Number of new category values is less or equal to {max_new}', condition)

    def add_condition_new_category_ratio_less_or_equal(self, max_ratio: float = 0):
        """Add condition - require column's ratio of instances with new categories to be less or equal to threshold.

        Parameters
        ----------
        max_ratio : float , default: 0
            Number of different categories value types which is the maximum allowed.
        """

        def condition(result: pd.DataFrame) -> ConditionResult:
            failing = result[result['Percent of new categories in sample'] > max_ratio][
                'Percent of new categories in sample'].apply(format_percent)
            if len(failing) > 0:
                return ConditionResult(ConditionCategory.FAIL,
                                       f'Found {len(failing)} columns with ratio '
                                       f'of new categories above threshold: \n{dict(failing)}')
            else:
                details = get_condition_passed_message(len(result))
                if any(result['Percent of new categories in sample'] > 0):
                    new_categories_columns = result[result['Percent of new categories in sample'] > 0]
                    new_categories_ratio = dict(new_categories_columns['Percent of new categories in sample']
                                                .apply(format_percent)[:5])
                    details += f'. Top columns with new categories ratio: \n{new_categories_ratio}'
                return ConditionResult(ConditionCategory.PASS, details)

        return self.add_condition(
            f'Ratio of samples with a new category is less or equal to {format_percent(max_ratio)}', condition)
