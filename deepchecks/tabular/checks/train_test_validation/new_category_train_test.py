# ----------------------------------------------------------------------------
# Copyright (C) 2021-2023 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""The new category train test check module."""
from typing import Dict, List, Optional, Union

import pandas as pd

from deepchecks.core import CheckResult, ConditionCategory, ConditionResult, DatasetKind
from deepchecks.core.fix_classes import TrainTestCheckFixMixin
from deepchecks.core.reduce_classes import ReduceFeatureMixin
from deepchecks.tabular import Context, TrainTestCheck
from deepchecks.tabular._shared_docs import docstrings
from deepchecks.tabular.utils.messages import get_condition_passed_message
from deepchecks.utils.dataframes import select_from_dataframe
from deepchecks.utils.strings import format_number, format_percent
from deepchecks.utils.typing import Hashable

__all__ = ['NewCategoryTrainTest']


@docstrings
class NewCategoryTrainTest(TrainTestCheck, ReduceFeatureMixin, TrainTestCheckFixMixin):
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
    aggregation_method: Optional[str], default: 'max'
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
            aggregation_method: Optional[str] = 'max',
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

            # Nans are not considered as new categories
            train_column = train_column.dropna()
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
                                   columns=['Feature Name',
                                            '# New Categories',
                                            'Ratio of New Categories',
                                            'New categories',
                                            'Feature importance']).set_index(['Feature Name'])
        result_data.sort_values(by='Ratio of New Categories', ascending=False, inplace=True)
        if all(feature_importance.isna()):
            result_data.drop('Feature importance', axis=1, inplace=True)

        if context.with_display:
            display = result_data.copy()
            display['Ratio of New Categories'] = display['Ratio of New Categories'].apply(format_percent)
            display['# New Categories'] = display['# New Categories'].apply(format_number)
            display['Examples'] = display['New categories']. \
                apply(lambda x: x[:self.max_new_categories_to_show])
            display.drop('New categories', axis=1, inplace=True)
            display = display.iloc[:self.max_features_to_show, :]
        else:
            display = None

        return CheckResult(result_data, display=display)

    def reduce_output(self, check_result: CheckResult) -> Dict[str, float]:
        """Return an aggregated drift score based on aggregation method defined."""
        feature_importance = check_result.value['Feature importance'] if 'Feature importance' \
                                                                         in check_result.value.columns else None
        values = check_result.value['Ratio of New Categories']
        return self.feature_reduce(self.aggregation_method, values, feature_importance, 'New Categories Ratio')

    def add_condition_new_categories_less_or_equal(self, max_new: int = 0):
        """Add condition - require column's number of different new categories to be less or equal to threshold.

        Parameters
        ----------
        max_new : int , default: 0
            Number of different categories value types which is the maximum allowed.
        """

        def condition(result: pd.DataFrame) -> ConditionResult:
            failing = result[result['# New Categories'] > max_new]['# New Categories']
            if len(failing) > 0:
                return ConditionResult(ConditionCategory.FAIL,
                                       f'Found {len(failing)} features with number'
                                       f' of new categories above threshold: \n{dict(failing)}')
            else:
                details = get_condition_passed_message(len(result), feature=True)
                if any(result['# New Categories'] > 0):
                    new_categories_columns = dict(
                        result[result['# New Categories'] > 0]['# New Categories'][:5])
                    details += f'. Top features with new categories count: \n{new_categories_columns}'
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
            failing = result[result['Ratio of New Categories'] > max_ratio][
                'Ratio of New Categories'].apply(format_percent)
            if len(failing) > 0:
                return ConditionResult(ConditionCategory.FAIL,
                                       f'Found {len(failing)} features with ratio '
                                       f'of new categories above threshold: \n{dict(failing)}')
            else:
                details = get_condition_passed_message(len(result), feature=True)
                if any(result['Ratio of New Categories'] > 0):
                    new_categories_columns = result[result['Ratio of New Categories'] > 0]
                    new_categories_ratio = dict(new_categories_columns['Ratio of New Categories']
                                                .apply(format_percent)[:5])
                    details += f'. Top features with new categories ratio: \n{new_categories_ratio}'
                return ConditionResult(ConditionCategory.PASS, details)

        return self.add_condition(
            f'Ratio of samples with a new category is less or equal to {format_percent(max_ratio)}', condition)

    def fix_logic(self, context: Context, check_result: CheckResult, fix_method='move_to_train',
                  max_ratio: float = 0, percentage_to_move: float = 0.5) -> Context:
        """Run fix.

        Parameters
        ----------
        context : Context
            Context object.
        check_result : CheckResult
            CheckResult object.
        fix_method : str, default: 'move_to_train'
            Method to fix the problem. Possible values: 'drop_features', 'replace_with_none', 'move_to_train'.
        max_ratio : float, default: 0
            Maximum ratio of samples with new categories.
        percentage_to_move : float, default: 0.5
            Percentage of samples with new categories to move to train.
        """
        train, test = context.train.data, context.test.data

        cols_to_fix = check_result.value[check_result.value['Ratio of New Categories'] > max_ratio].index

        if fix_method == 'drop_features':
            train = train.drop(columns=cols_to_fix)
            test = test.drop(columns=cols_to_fix)
        elif fix_method == 'replace_with_none':
            for col in cols_to_fix:
                new_categories = check_result.value['New categories'][col]
                train[col] = train[col].apply(lambda x: None if x in new_categories else x)
                test[col] = test[col].apply(lambda x: None if x in new_categories else x)
        elif fix_method == 'move_to_train':
            # The following code takes the samples with new categories and moves 0.5 of them from test to train:
            for col in cols_to_fix:
                new_categories = check_result.value['New categories'][col]
                new_categories_train = test[test[col].isin(new_categories)].sample(frac=percentage_to_move)
                train = train.append(new_categories_train)
                test = test.drop(new_categories_train.index)
        else:
            raise ValueError(f'Fix method {fix_method} is not supported')

        context.set_dataset_by_kind(DatasetKind.TRAIN, context.train.copy(train))
        context.set_dataset_by_kind(DatasetKind.TEST, context.test.copy(test))

        return context

    @property
    def fix_params(self):
        """Return fix params for display."""
        return {'fix_method': {'display': 'Fix By',
                               'params': ['drop_features', 'replace_with_nones', 'move_to_train'],
                               'params_display': ['Dropping Features', 'Replacing With Nones',
                                                  'Moving Samples To Train'],
                               'params_description': ['Drop features with new categories from the dataset',
                                                      'Replace new categories with None values',
                                                      'Move samples with new categories from test to train']},
                'max_ratio': {'display': 'Max Ratio Of New Categories',
                              'params': float,
                              'params_display': 0.0,
                              'params_description': 'Maximum ratio of samples with new categories'},
                'percentage_to_move': {'display': 'Percentage Of Samples To Move',
                                       'params': float,
                                       'params_display': 0.2,
                                       'params_description': 'Percentage of samples with new categories to move to '
                                                             'train. Relevant only for move_to_train fix method.'}}

    @property
    def problem_description(self):
        """Return problem description."""
        return """New categories are present in test data but not in train data. This can lead to wrong predictions in 
                  test data, as these new categories were unknown to the model during training."""

    @property
    def manual_solution_description(self):
        """Return manual solution description."""
        return """Resample train data to include samples with these new categories."""

    @property
    def automatic_solution_description(self):
        """Return automatic solution description."""
        return """There are 4 possible automatic solutions:
                  1. Drop features with new categories from the dataset, so the model won't train on them
                  3. Replace new categories with None values, so the model will treat them as missing values. 
                     This is not recommended, as it doesn't solve the underlying issue.
                  4. Move samples with new categories from test to train. 
                     This is the recommended solution, as it allows the model to be trained on the correct data. 
                     However, this solution can cause data leakage, and should only be used if it's possible for train 
                     dataset to have samples from test. 
                     For example, if train and test datasets are from 2 different time periods, it would probably be 
                     considered leakage if test samples (from the later time period) were moved to train."""
