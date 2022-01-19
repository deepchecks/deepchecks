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
from typing import List
import pandas as pd

from deepchecks import Dataset
from deepchecks.base.check import CheckResult, ConditionResult, TrainTestBaseCheck
from deepchecks.utils.strings import format_percent
from deepchecks.utils.typing import Hashable


pd.options.mode.chained_assignment = None


__all__ = ['TrainTestSamplesMix']


class TrainTestSamplesMix(TrainTestBaseCheck):
    """Detect samples in the test data that appear also in training data."""

    def run(self, train_dataset: Dataset, test_dataset: Dataset,  model=None) -> CheckResult:
        """Run check.

        Args:
            train_dataset (Dataset): The training dataset object. Must contain an index.
            test_dataset (Dataset): The test dataset object. Must contain an index.
            model (): any = None - not used in the check

        Returns:
            CheckResult: value is sample leakage ratio in %,
            displays a dataframe that shows the duplicated rows between the datasets

        Raises:
            DeepchecksValueError: If the object is not a Dataset instance
        """
        return self._data_sample_leakage_report(test_dataset=test_dataset, train_dataset=train_dataset)

    def _data_sample_leakage_report(self, test_dataset: Dataset, train_dataset: Dataset):
        test_dataset = Dataset.ensure_not_empty_dataset(test_dataset, cast=True)
        train_dataset = Dataset.ensure_not_empty_dataset(train_dataset, cast=True)
        features = self._datasets_share_features([test_dataset, train_dataset])
        label_name = self._datasets_share_label([test_dataset, train_dataset])

        columns = features + [label_name]

        # For pandas.groupby in python 3.6, there is problem with comparing numpy nan, so replace with placeholder
        na_filler = '__deepchecks_na_filler__'
        train_df = train_dataset.data.fillna(value=na_filler)
        test_df = test_dataset.data.fillna(value=na_filler)

        train_uniques = _create_unique_frame(train_df, columns, text_prefix='Train indices: ')
        test_uniques = _create_unique_frame(test_df, columns, text_prefix='Test indices: ')

        duplicates_df, test_dup_count = _create_train_test_joined_duplicate_frame(train_uniques, test_uniques, columns)

        # Replace filler back to none
        duplicates_df = duplicates_df.applymap(lambda x: None if x == na_filler else x)
        dup_ratio = test_dup_count / test_dataset.n_samples
        user_msg = f'{format_percent(dup_ratio)} ({test_dup_count} / {test_dataset.n_samples}) \
                     of test data samples appear in train data'
        display = [user_msg, duplicates_df.head(10)] if dup_ratio else None
        result = {'ratio': dup_ratio, 'data': duplicates_df}
        return CheckResult(result, header='Train Test Samples Mix', display=display)

    def add_condition_duplicates_ratio_not_greater_than(self, max_ratio: float = 0.1):
        """Add condition - require max allowed ratio of test data samples to appear in train data.

        Args:
            max_ratio (float): Max allowed ratio of test data samples to appear in train data
        """
        def condition(result: dict) -> ConditionResult:
            ratio = result['ratio']
            if ratio > max_ratio:
                return ConditionResult(False,
                                       f'Percent of test data samples that appear in train data: '
                                       f'{format_percent(ratio)}')
            else:
                return ConditionResult(True)

        return self.add_condition(f'Percentage of test data samples that appear in train data '
                                  f'not greater than {format_percent(max_ratio)}',
                                  condition)


def _create_train_test_joined_duplicate_frame(first: pd.DataFrame, second: pd.DataFrame, columns: List[Hashable]):
    """Create duplicate dataframe out of 2 uniques dataframes.

    This function accept 2 dataframes resulted from `_create_unique_frame`. this means that each dataframe have
    no duplicate in it. so if the concatenation between the 2 find duplicates, they are necessarily between each other.
    """
    columns_data = []
    index_text = []
    total_test_count = 0
    group_unique_data: dict = pd.concat([first, second]).groupby(columns, dropna=False).groups
    # The group data is backward (the columns are the indexes, and the indexes are the values)
    for duplicate_columns, indexes in group_unique_data.items():
        # If length is 1, then no duplicate found between first and second
        if len(indexes) == 1:
            continue
        # Indexes should have dict of train & test info from `_filter_duplicates`
        text = indexes[0]['text'] + '\n' + indexes[1]['text']
        # Take the count only of test
        test_count = indexes[0]['count'] if indexes[0]['text'].startswith('Test') else indexes[1]['count']
        total_test_count += test_count
        # Save info of duplicated text and the columns info
        columns_data.append([*duplicate_columns, test_count])
        index_text.append(text)

    count_column_name = '_value_to_sort_by_'
    duplicates = pd.DataFrame(columns_data, index=index_text, columns=[*columns, count_column_name])
    duplicates = duplicates.sort_values(by=count_column_name, ascending=False)
    duplicates = duplicates.drop(count_column_name, axis=1)
    return duplicates, total_test_count


def _create_unique_frame(df: pd.DataFrame, columns: List[Hashable], text_prefix: str = '') -> pd.DataFrame:
    """For given dataframe and columns create a dataframe with only unique combinations of the columns."""
    columns_data = []
    index_text = []
    group_unique_data: dict = df.groupby(columns, dropna=False).groups
    # The group data is backward (the columns are the indexes, and the indexes are the values)
    for duplicate_columns, indexes in group_unique_data.items():
        # Save info of duplicated text and the columns info
        columns_data.append(duplicate_columns)
        index_text.append(_get_dup_info(indexes, text_prefix))

    return pd.DataFrame(columns_data, index=index_text, columns=columns)


def _get_dup_info(index_arr: list, text_prefix: str) -> dict:
    text = ', '.join([str(i) for i in index_arr])
    if len(text) > 30:
        text = f'{text[:30]}.. Tot. {(len(index_arr))}'

    return {'text': f'{text_prefix}{text}', 'count': len(index_arr)}
