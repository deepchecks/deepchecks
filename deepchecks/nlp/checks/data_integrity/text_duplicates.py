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
"""Module contains Data Duplicates check."""
import typing as t

import pandas as pd

from deepchecks.core import CheckResult
from deepchecks.nlp import Context, SingleDatasetCheck
from deepchecks.nlp._shared_docs import docstrings
from deepchecks.nlp.text_data import TextData
from deepchecks.nlp.utils.text import cut_string, hash_samples, normalize_samples
from deepchecks.utils.abstracts.data_duplicates import DataDuplicatesAbstract
from deepchecks.utils.dataframes import hide_index_for_display
from deepchecks.utils.other import to_ordional_enumeration
from deepchecks.utils.strings import format_list, format_percent, truncate_string

__all__ = ['TextDuplicates']


@docstrings
class TextDuplicates(SingleDatasetCheck, DataDuplicatesAbstract):
    """Checks for duplicate samples in the dataset.

    Parameters
    ----------
    {text_normalization_params:1*indent}
    n_to_show : int, default: 5
        number of most common duplicated samples to show.
    n_samples : int, default: 10_000_000
        number of samples to use for this check.
    random_state : int, default: 42
        random seed for all check internals.
    {max_text_length_for_display_param:1*indent}
    """

    def __init__(
        self,
        ignore_case: bool = True,
        remove_punctuation: bool = True,
        normalize_unicode: bool = True,
        remove_stopwords: bool = True,
        ignore_whitespace: bool = False,
        n_to_show: int = 5,
        n_samples: int = 10_000_000,
        random_state: int = 42,
        max_text_length_for_display: int = 30,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.ignore_whitespace = ignore_whitespace
        self.ignore_case = ignore_case
        self.remove_punctuation = remove_punctuation
        self.normalize_unicode = normalize_unicode
        self.remove_stopwords = remove_stopwords
        self.n_to_show = n_to_show
        self.n_samples = n_samples
        self.random_state = random_state
        self.max_text_length_for_display = max_text_length_for_display

    @property
    def _text_normalization_kwargs(self):
        return {
            'ignore_case': self.ignore_case,
            'ignore_whitespace': self.ignore_whitespace,
            'normalize_uni': self.normalize_unicode,
            'remove_punct': self.remove_punctuation,
            'remove_stops': self.remove_stopwords,
        }

    def _truncate_text(self, x: str) -> str:
        return truncate_string(x, self.max_text_length_for_display)

    def _create_df(self, dataset, samples):
        sample_hashes = hash_samples(normalize_samples(samples, **self._text_normalization_kwargs))
        df = pd.DataFrame({
            'Text': samples,
            'hash': sample_hashes,
            'Sample ID': dataset.get_original_text_indexes()
        })
        return df

    def run_logic(self, context: Context, dataset_kind):
        """Run check."""
        dataset = context.get_data_by_kind(dataset_kind).sample(self.n_samples, random_state=self.random_state)
        dataset = t.cast(TextData, dataset)
        n_of_unique = 0
        n_of_samples = len(dataset)

        # First run logic on truncated samples to speed up computation
        truncated_samples = [cut_string(x) for x in dataset.text]
        df_truncated = self._create_df(dataset, truncated_samples)

        grouped_samples_truncated = df_truncated.groupby(by=['hash'], dropna=False, group_keys=True)
        reinspect_idx = df_truncated[grouped_samples_truncated['Text'].transform('count') > 1].index.to_list()
        # At this stage, what was detected as unique is actually unique
        n_of_unique += sum(grouped_samples_truncated['Text'].transform('count') == 1)

        # Reinspect samples that are truncated
        dataset = dataset.copy(rows_to_use=reinspect_idx)
        if len(dataset) == 0:
            return CheckResult(value={'percent_of_duplicates': 0,
                                      'duplicates': pd.DataFrame()})

        samples = dataset.text

        df = self._create_df(dataset, samples)
        grouped_samples = df.groupby(by=['hash'], dropna=False)
        counted_samples = grouped_samples['Text'].size()
        # Once we arrived here (inspecting only samples suspected to be duplicates), we can add them to the count
        n_of_unique += len(counted_samples)
        percent_of_duplicates = 1 - n_of_unique / n_of_samples

        counted_duplicates = counted_samples[counted_samples > 1]
        duplicates_hashes = set(counted_duplicates.index)

        result_df = df[df['hash'].isin(duplicates_hashes)]
        result_df = result_df.rename(columns={'hash': 'Duplicate'})
        duplicates_enumeration = to_ordional_enumeration(result_df['Duplicate'].to_list())
        result_df['Duplicate'] = result_df['Duplicate'].apply(lambda x: duplicates_enumeration[x])
        result_df = result_df.set_index(['Duplicate', 'Sample ID'])

        result_value = {
            'percent_of_duplicates': percent_of_duplicates,
            'duplicates': result_df
        }

        if context.with_display is False or percent_of_duplicates == 0:
            return CheckResult(value=result_value)

        grouped_samples = df[df['hash'].isin(duplicates_hashes)].groupby(by=['hash'], dropna=False)
        first_sample = grouped_samples['Text'].first()
        sample_ids = grouped_samples['Sample ID'].aggregate(lambda x: format_list(x.to_list()))

        table = pd.DataFrame({
            'Text': first_sample.apply(self._truncate_text),
            'Sample IDs': sample_ids,
            'Number of Samples': counted_duplicates
        })
        table = table.iloc[:self.n_to_show]
        table = table.sort_values(by=['Number of Samples'], ascending=False)

        return CheckResult(
            value=result_value,
            display=[
                f'{format_percent(percent_of_duplicates)} of data samples are duplicates.',
                'Each row in the table shows an example of a text duplicate and the number of times it appears.',
                hide_index_for_display(table)
            ]
        )
