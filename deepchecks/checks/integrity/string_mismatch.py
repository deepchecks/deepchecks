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
import itertools

import pandas as pd

from deepchecks.base.check_context import CheckRunContext
from deepchecks import (
    CheckResult,
    SingleDatasetBaseCheck,
    ConditionResult,
    ConditionCategory
)
from deepchecks.utils.dataframes import select_from_dataframe
from deepchecks.utils.features import N_TOP_MESSAGE, column_importance_sorter_df
from deepchecks.utils.typing import Hashable
from deepchecks.utils.strings import (
    get_base_form_to_variants_dict,
    is_string_column,
    format_percent
)


__all__ = ['StringMismatch']


class StringMismatch(SingleDatasetBaseCheck):
    """Detect different variants of string categories (e.g. "mislabeled" vs "mis-labeled") in a categorical column.

    Parameters
    ----------
    columns : Union[Hashable, List[Hashable]] , default: None
        Columns to check, if none are given checks all columns except ignored ones.
    ignore_columns : Union[Hashable, List[Hashable]] , default: None
        Columns to ignore, if none given checks based on columns variable
    n_top_columns : int , optional
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

    def run_logic(self, context: CheckRunContext, dataset_type: str = 'train') -> CheckResult:
        """Run check."""
        if dataset_type == 'train':
            dataset = context.train
        else:
            dataset = context.test

        df = select_from_dataframe(dataset.data, self.columns, self.ignore_columns)

        results = []
        result_dict = defaultdict(dict)

        for column_name in df.columns:
            column: pd.Series = df[column_name]
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
            df_graph = column_importance_sorter_df(df_graph, dataset, context.features_importance,
                                                   self.n_top_columns, col='Column Name')
            display = [N_TOP_MESSAGE % self.n_top_columns, df_graph]
        else:
            display = None

        return CheckResult(result_dict, display=display)

    def add_condition_not_more_variants_than(self, num_max_variants: int):
        """Add condition - no more than given number of variants are allowed (per string baseform).

        Parameters
        ----------
        num_max_variants : int
            Maximum number of variants allowed.
        """
        name = f'Not more than {num_max_variants} string variants'
        return self.add_condition(name, _condition_variants_number, num_max_variants=num_max_variants)

    def add_condition_no_variants(self):
        """Add condition - no variants are allowed."""
        name = 'No string variants'
        return self.add_condition(name, _condition_variants_number, num_max_variants=0)

    def add_condition_ratio_variants_not_greater_than(self, max_ratio: float = 0.01):
        """Add condition - percentage of variants in data is not allowed above given threshold.

        Parameters
        ----------
        max_ratio : float , default: 0.01
            Maximum percent of variants allowed in data.
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
                details = f'Found columns with variants ratio above threshold: {not_passing_columns}'
                return ConditionResult(False, details)
            return ConditionResult(True)

        name = f'Ratio of variants is not greater than {format_percent(max_ratio)}'
        return self.add_condition(name, condition, max_ratio=max_ratio)


def _condition_variants_number(result, num_max_variants: int, max_cols_to_show: int = 5, max_forms_to_show: int = 5):
    not_passing_variants = defaultdict(list)
    for col, baseforms in result.items():
        for base_form, variants_list in baseforms.items():
            if len(variants_list) > num_max_variants:
                if len(not_passing_variants[col]) < max_forms_to_show:
                    not_passing_variants[col].append(base_form)
    if not_passing_variants:
        variants_to_show = dict(itertools.islice(not_passing_variants.items(), max_cols_to_show))
        details = f'Found columns with amount of variants above threshold: {variants_to_show}'
        return ConditionResult(False, details, ConditionCategory.WARN)
    return ConditionResult(True)
