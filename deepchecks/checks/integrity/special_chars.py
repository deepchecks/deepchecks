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
"""module contains Invalid Chars check."""
from collections import defaultdict
from typing import Union, List
import pandas as pd
from pandas.api.types import infer_dtype

from deepchecks import Dataset, CheckResult, SingleDatasetBaseCheck, ConditionResult, ConditionCategory
from deepchecks.utils.dataframes import select_from_dataframe
from deepchecks.utils.features import N_TOP_MESSAGE, calculate_feature_importance_or_none, column_importance_sorter_df
from deepchecks.utils.strings import string_baseform, format_percent
from deepchecks.utils.validation import ensure_dataframe_type
from deepchecks.utils.typing import Hashable


__all__ = ['SpecialCharacters']


class SpecialCharacters(SingleDatasetBaseCheck):
    """Search in column[s] for values that contains only special characters.

    Args:
        columns (Union[Hashable, List[Hashable]]):
            Columns to check, if none are given checks all columns except ignored ones.
        ignore_columns (Union[Hashable, List[Hashable]]):
            Columns to ignore, if none given checks based on columns variable.
        n_most_common (int):
            Number of most common special-only samples to show in results
        n_top_columns (int): (optional - used only if model was specified)
          amount of columns to show ordered by feature importance (date, index, label are first)
    """

    def __init__(
        self,
        columns: Union[Hashable, List[Hashable], None] = None,
        ignore_columns: Union[Hashable, List[Hashable], None] = None,
        n_most_common: int = 2,
        n_top_columns: int = 10
    ):
        super().__init__()
        self.columns = columns
        self.ignore_columns = ignore_columns
        self.n_most_common = n_most_common
        self.n_top_columns = n_top_columns

    def run(self, dataset, model=None) -> CheckResult:
        """Run check.

        Args:
          dataset(Dataset):

        Returns:
          (CheckResult): DataFrame with ('invalids') for any column with special_characters chars.
        """
        feature_importances = calculate_feature_importance_or_none(model, dataset)
        return self._special_characters(dataset, feature_importances)

    def _special_characters(self, dataset: Union[pd.DataFrame, Dataset],
                            feature_importances: pd.Series = None) -> CheckResult:
        """Run check.

        Args:
            dataset (Dataset): a Dataset or DataFrame object.
        Returns:
            (CheckResult): DataFrame with columns ('Column Name', '% Invalid Samples', 'Most Common Invalids Samples')
              for any column that contains invalid chars.
        """
        # Validate parameters
        dataset: pd.DataFrame = ensure_dataframe_type(dataset)
        dataset = select_from_dataframe(dataset, self.columns, self.ignore_columns)

        # Result value: { Column Name: {invalid: pct}}
        display_array = []
        result = {}

        for column_name in dataset.columns:
            column_data = dataset[column_name]
            # Get dict of samples to count
            special_samples = _get_special_samples(column_data)
            if special_samples:
                result[column_name] = sum(special_samples.values()) / column_data.size
                percent = format_percent(sum(special_samples.values()) / column_data.size)
                top_n_samples_items = \
                    sorted(special_samples.items(), key=lambda x: x[1], reverse=True)[:self.n_most_common]
                top_n_samples_values = [item[0] for item in top_n_samples_items]
                display_array.append([column_name, percent, top_n_samples_values])

        df_graph = pd.DataFrame(display_array,
                                columns=['Column Name', '% Special-Only Samples', 'Most Common Special-Only Samples'])
        df_graph = df_graph.set_index(['Column Name'])
        df_graph = column_importance_sorter_df(df_graph, dataset, feature_importances,
                                               self.n_top_columns, col='Column Name')
        display = [N_TOP_MESSAGE % self.n_top_columns, df_graph] if len(df_graph) > 0 else None

        return CheckResult(result, display=display)

    def add_condition_ratio_of_special_characters_not_grater_than(self, max_ratio: float = 0.001):
        """Add condition - ratio of entirely special character in column.

        Args:
            max_ratio(float): Maximum ratio allowed.
        """
        name = f'Ratio of entirely special character samples not greater '\
               f'than {format_percent(max_ratio)}'

        def condition(result):
            not_passed = {}
            if result:
                for column_name, ratio in result.items():
                    if ratio > max_ratio:
                        not_passed[column_name] = format_percent(ratio)

            if not_passed:
                return ConditionResult(False, f'Found columns with ratio above threshold: {not_passed}',
                                       category=ConditionCategory.WARN)
            return ConditionResult(True)

        return self.add_condition(name, condition)


def _get_special_samples(column_data: pd.Series) -> Union[dict, None]:
    if not _is_stringed_type(column_data):
        return None
    samples_to_count = defaultdict(lambda: 0)
    for sample in column_data:
        if isinstance(sample, str) and len(sample) > 0 and len(string_baseform(sample)) == 0:
            samples_to_count[sample] = samples_to_count[sample] + 1

    return samples_to_count or None


def _is_stringed_type(col) -> bool:
    return infer_dtype(col) not in ['integer', 'decimal', 'floating']
