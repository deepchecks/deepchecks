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
"""String mismatch functions."""
from typing import List, Union

import pandas as pd
from merge_args import merge_args

from deepchecks.core import CheckResult, ConditionCategory, ConditionResult, DatasetKind
from deepchecks.core.fix_classes import TrainTestCheckFixMixin, FixResult
from deepchecks.tabular import Context, TrainTestCheck
from deepchecks.tabular._shared_docs import docstrings
from deepchecks.tabular.utils.feature_importance import N_TOP_MESSAGE, column_importance_sorter_df
from deepchecks.tabular.utils.messages import get_condition_passed_message
from deepchecks.utils.dataframes import select_from_dataframe
from deepchecks.utils.strings import format_percent, get_base_form_to_variants_dict, is_string_column
from deepchecks.utils.typing import Hashable

__all__ = ['StringMismatchComparison']


class StringMismatchComparison(TrainTestCheck, TrainTestCheckFixMixin):
    """Detect different variants of string categories between the same categorical column in two datasets.

    This check compares the same categorical column within a dataset and baseline and checks whether there are
    variants of similar strings that exists only in dataset and not in baseline.
    Specifically, we define similarity between strings if they are equal when ignoring case and non-letter
    characters.
    Example:
    We have a train dataset with similar strings 'string' and 'St. Ring', which have different meanings.
    Our tested dataset has the strings 'string', 'St. Ring' and a new phrase, 'st.  ring'.
    Here, we have a new variant of the above strings, and would like to be acknowledged, as this is obviously a
    different version of 'St. Ring'.

    Parameters
    ----------
    columns : Union[Hashable, List[Hashable]] , default: None
        Columns to check, if none are given checks all columns except ignored ones.
    ignore_columns : Union[Hashable, List[Hashable]] , default: None
        Columns to ignore, if none given checks based on columns variable
    n_top_columns : int , optional
        amount of columns to show ordered by feature importance (date, index, label are first)
    n_samples : int , default: 1_000_000
        number of samples to use for this check.
    random_state : int, default: 42
        random seed for all check internals.
    """

    def __init__(
        self,
        columns: Union[Hashable, List[Hashable], None] = None,
        ignore_columns: Union[Hashable, List[Hashable], None] = None,
        n_top_columns: int = 10,
        n_samples: int = 1_000_000,
        random_state: int = 42,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.columns = columns
        self.ignore_columns = ignore_columns
        self.n_top_columns = n_top_columns
        self.n_samples = n_samples
        self.random_state = random_state

    def run_logic(self, context: Context) -> CheckResult:
        """Run check.

        Returns
        -------
        CheckResult
            with value of type dict that contains detected different variants of string
        """
        # Validate parameters
        df = context.test.sample(self.n_samples, random_state=self.random_state).data
        df = select_from_dataframe(df, self.columns, self.ignore_columns)
        baseline_df = context.train.sample(self.n_samples, random_state=self.random_state).data

        display_mismatches = []
        result_dict = {}

        # Get shared columns
        columns = set(df.columns).intersection(baseline_df.columns)

        for column_name in columns:
            tested_column: pd.Series = df[column_name]
            baseline_column: pd.Series = baseline_df[column_name]
            # If one of the columns isn't string type, continue
            if not is_string_column(tested_column) or not is_string_column(baseline_column):
                continue

            result_dict[column_name] = {}
            tested_counts = tested_column.value_counts()
            baseline_counts = baseline_column.value_counts()
            tested_baseforms = get_base_form_to_variants_dict(tested_column.unique())
            baseline_baseforms = get_base_form_to_variants_dict(baseline_column.unique())

            common_baseforms = set(tested_baseforms.keys()).intersection(baseline_baseforms.keys())
            for baseform in common_baseforms:
                tested_values = tested_baseforms[baseform]
                baseline_values = baseline_baseforms[baseform]
                # If at least one unique value in tested dataset, add the column to results
                if len(tested_values - baseline_values) > 0:
                    # Calculate all values to be shown
                    variants_only_in_dataset = list(tested_values - baseline_values)
                    variants_only_in_baseline = list(baseline_values - tested_values)
                    common_variants = list(tested_values & baseline_values)
                    percent_variants_only_in_dataset = _percentage_in_series(tested_column, tested_counts,
                                                                             variants_only_in_dataset)
                    percent_variants_in_baseline = _percentage_in_series(baseline_column, baseline_counts,
                                                                         variants_only_in_baseline)
                    result_dict[column_name][baseform] = {
                        'commons': common_variants, 'variants_only_in_test': variants_only_in_dataset,
                        'variants_only_in_train': variants_only_in_baseline,
                        'percent_variants_only_in_test': percent_variants_only_in_dataset[0],
                        'percent_variants_in_train': percent_variants_in_baseline[0]
                    }
                    if context.with_display:
                        display_mismatches.append([column_name, baseform, common_variants,
                                                   variants_only_in_dataset, percent_variants_only_in_dataset[1],
                                                   variants_only_in_baseline, percent_variants_in_baseline[1]])

        # Create result dataframe
        if display_mismatches:
            df_graph = pd.DataFrame(display_mismatches,
                                    columns=['Column name', 'Base form', 'Common variants', 'Variants only in test',
                                             '% Unique variants out of all dataset samples (count)',
                                             'Variants only in train',
                                             '% Unique variants out of all baseline samples (count)'])
            df_graph = df_graph.set_index(['Column name', 'Base form'])

            df_graph = column_importance_sorter_df(
                df_graph,
                context.test,
                context.feature_importance,
                self.n_top_columns,
                col='Column name'
            )
            # For display transpose the dataframe
            display = [N_TOP_MESSAGE % self.n_top_columns, df_graph.T]
        else:
            display = None

        return CheckResult(result_dict, display=display)

    def add_condition_no_new_variants(self):
        """Add condition - no new variants allowed in test data."""
        name = 'No new variants allowed in test data'
        return self.add_condition(name, _condition_percent_limit, ratio=0)

    def add_condition_ratio_new_variants_less_or_equal(self, ratio: float):
        """Add condition - require new variants' percentage in test data to be less or equal to the threshold.

        Parameters
        ----------
        ratio : float
            Max percentage of new variants in test data allowed.
        """
        name = f'Ratio of new variants in test data is less or equal to {format_percent(ratio)}'
        return self.add_condition(name, _condition_percent_limit, ratio=ratio)

    @docstrings
    @merge_args(TrainTestCheck.run)
    def fix(self, *args, check_result: CheckResult = None, **kwargs) -> FixResult:
        """Run fix.

        Parameters
        ----------
        {additional_context_params:2*indent}
        check_result : CheckResult, default: None
            CheckResult object to use for fixing the dataset.

        Returns
        -------
        Dataset
            Dataset with fixed duplicates."""
        context = self.get_context(*args, **kwargs)

        if check_result is None:
            check_result = self.run_logic(context)

        train, test = context.train, context.test
        train_data, test_data = train.data, test.data

        for col, variants in check_result.value.items():
            # We take the most common form from train data:
            value_counts = train_data[col].value_counts()
            for baseform, details in variants.items():
                all_variants = set(details['variants_only_in_train'] + details['commons'] + details['variants_only_in_test'])

                train_variants = sorted([variant for variant in all_variants if variant in value_counts.index])
                most_common_variant = value_counts[train_variants].sort_values(ascending=False).index[0]

                # most_common_variant = sorted([(var['variant'], var['percent']) for var in details],
                #                              key=lambda x: x[1], reverse=True)[0][0]
                train_data[col] = train_data[col].apply(lambda x: most_common_variant if x in all_variants else x)
                test_data[col] = test_data[col].apply(lambda x: most_common_variant if x in all_variants else x)

        return FixResult(fixed_train=train.copy(train_data), fixed_test=test.copy(test_data))


def _condition_percent_limit(result, ratio: float):
    not_passing_columns = {}
    for col, baseforms in result.items():
        sum_percent = 0
        for info in baseforms.values():
            sum_percent += info['percent_variants_only_in_test']
        if sum_percent > ratio:
            not_passing_columns[col] = format_percent(sum_percent)

    if not_passing_columns:
        details = f'Found {len(not_passing_columns)} out of {len(result)} relevant columns with ratio of variants ' \
                  f'above threshold: {not_passing_columns}'
        return ConditionResult(ConditionCategory.FAIL, details)
    return ConditionResult(ConditionCategory.PASS, get_condition_passed_message(result))


def _percentage_in_series(series, counts, values):
    count = sum([counts[value] for value in values])
    percent = count / series.size
    return percent, f'{format_percent(percent)} ({count})'
