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
"""String mismatch functions."""
from collections import defaultdict
from typing import Union, List

import pandas as pd

from deepchecks import (
    CheckResult,
    SingleDatasetBaseCheck,
    Dataset,
    ensure_dataframe_type,
    ConditionResult,
    ConditionCategory
)
from deepchecks.utils.dataframes import filter_columns_with_validation
from deepchecks.utils.features import calculate_feature_importance_or_null, column_importance_sorter_df
from deepchecks.utils.typing import Hashable
from deepchecks.utils.strings import (
    get_base_form_to_variants_dict,
    is_string_column,
    format_percent,
    format_columns_for_condition
)


__all__ = ['StringMismatch']


def _condition_variants_number(result, num_max_variants: int):
    not_passing_variants = defaultdict(list)
    for col, baseforms in result.items():
        for base_form, variants_list in baseforms.items():
            if len(variants_list) > num_max_variants:
                not_passing_variants[col].append(base_form)
    if not_passing_variants:
        details = f'Found columns with variants: {dict(not_passing_variants)}'
        return ConditionResult(False, details, ConditionCategory.WARN)
    return ConditionResult(True)


class StringMismatch(SingleDatasetBaseCheck):
    """Detect different variants of string categories (e.g. "mislabeled" vs "mis-labeled") in a categorical column.

    Args:
        columns (Union[Hashable, List[Hashable]]):
            Columns to check, if none are given checks all columns except ignored ones.
        ignore_columns (Union[Hashable, List[Hashable]]):
            Columns to ignore, if none given checks based on columns variable
        n_top_columns (int): (optinal - used only if model was specified)
          amount of columns to show ordered by feature importance (date, index, label are first)
    """

    def __init__(
        self,
        columns: Union[Hashable, List[Hashable], None] = None,
        ignore_columns: Union[Hashable, List[Hashable], None] = None,
        n_top_columns: int = 10
    ):
        super().__init__()
        self.columns = columns
        self.ignore_columns = ignore_columns
        self.n_top_columns = n_top_columns

    def run(self, dataset, model=None) -> CheckResult:
        """Run check.

        Args:
            dataset (DataFrame): A dataset or pd.FataFrame object.
        """
        feature_importances = calculate_feature_importance_or_null(dataset, model)
        return self._string_mismatch(dataset, feature_importances)

    def _string_mismatch(self, dataset: Union[pd.DataFrame, Dataset],
                         feature_importances: pd.Series=None) -> CheckResult:
        # Validate parameters
        original_dataset = dataset
        dataset: pd.DataFrame = ensure_dataframe_type(dataset)
        dataset = filter_columns_with_validation(dataset, self.columns, self.ignore_columns)

        results = []
        result_dict = defaultdict(dict)

        for column_name in dataset.columns:
            column: pd.Series = dataset[column_name]
            if not is_string_column(column):
                continue

            uniques = column.unique()
            base_form_to_variants = get_base_form_to_variants_dict(uniques)
            for base_form, variants in base_form_to_variants.items():
                if len(variants) == 1:
                    continue
                result_dict[column_name][base_form] = []
                for variant in variants:
                    count = sum(column == variant)
                    percent = count / len(column)
                    results.append([column_name, base_form, variant, count, format_percent(percent)])
                    result_dict[column_name][base_form].append({
                        'variant': variant, 'count': count, 'percent': percent
                    })

        # Create dataframe to display graph
        if results:
            df_graph = pd.DataFrame(results, columns=['Column Name', 'Base form', 'Value', 'Count', '% In data'])
            df_graph = df_graph.set_index(['Column Name', 'Base form'])
            df_graph = column_importance_sorter_df(df_graph, original_dataset, feature_importances,
                                                   self.n_top_columns, col='Column Name')
            display = df_graph
        else:
            display = None

        return CheckResult(result_dict, display=display)

    def add_condition_not_more_variants_than(self, num_max_variants: int):
        """Add condition - no more than given number of variants are allowed (per string baseform).

        Args:
            num_max_variants (int): Maximum number of variants allowed.
        """
        column_names = format_columns_for_condition(self.columns, self.ignore_columns)
        name = f'Not more than {num_max_variants} string variants for {column_names}'
        return self.add_condition(name, _condition_variants_number, num_max_variants=num_max_variants)

    def add_condition_no_variants(self):
        """Add condition - no variants are allowed."""
        column_names = format_columns_for_condition(self.columns, self.ignore_columns)
        name = f'No string variants for {column_names}'
        return self.add_condition(name, _condition_variants_number, num_max_variants=0)

    def add_condition_ratio_variants_not_more_than(self, max_ratio: float = 0.01):
        """Add condition - percentage of variants in data is not allowed above given threshold.

        Args:
            max_ratio (float): Maximum percent of variants allowed in data.
        """
        def condition(result, max_ratio: float):
            not_passing_columns = {}
            for col, baseforms in result.items():
                variants_percent_sum = 0
                for variants_list in baseforms.values():
                    variants_percent_sum += sum([v['percent'] for v in variants_list])
                if variants_percent_sum > max_ratio:
                    not_passing_columns[col] = format_percent(variants_percent_sum)

            if not_passing_columns:
                details = f'Found columns with variants ratio: {not_passing_columns}'
                return ConditionResult(False, details)
            return ConditionResult(True)

        column_names = format_columns_for_condition(self.columns, self.ignore_columns)
        name = f'Not more than {format_percent(max_ratio)} variants for {column_names}'
        return self.add_condition(name, condition, max_ratio=max_ratio)
