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
"""Module contains Frequent Substrings check."""
import typing as t
from collections import defaultdict
from typing import Dict

import pandas as pd

from deepchecks.core import CheckResult, ConditionCategory, ConditionResult
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.nlp import Context, SingleDatasetCheck
from deepchecks.nlp._shared_docs import docstrings
from deepchecks.nlp.text_data import TextData
from deepchecks.utils.strings import format_percent

__all__ = ['FrequentSubstrings']


@docstrings
class FrequentSubstrings(SingleDatasetCheck):
    """Checks for frequent substrings in the dataset.

    Parameters
    ----------
    n_to_show : int, default: 5
        number of most common duplicated samples to show.
    n_samples : int, default: 10_000_000
        number of samples to use for this check.
    random_state : int, default: 42
        random seed for all check internals.
    min_threshold: float, default: 0.05
        minimum threshold for tagging a substring as frequent
    min_ngram: int, default: 4
        minimum n for the n-gram extraction
    epsilon: float, default: 0.02
        maximum difference for filtering long substrings
    significant_threshold: float, default: 0.3
        all the samples above the frequency would be returned

    """

    def __init__(
        self,
        n_to_show: int = 5,
        n_samples: int = 10_000_000,
        random_state: int = 42,
        min_threshold: float = 0.05,
        min_ngram: int = 4,
        epsilon=0.02,
        significant_threshold: float = 0.3,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_to_show = n_to_show
        self.n_samples = n_samples
        self.random_state = random_state
        self.min_threshold = min_threshold
        self.min_ngram = min_ngram
        self.epsilon = epsilon
        self.significant_threshold = significant_threshold

    @staticmethod
    def _extract_ngrams(text, n):
        words = text.split()
        return [' '.join(words[i:i + n]) for i in range(len(words) - n + 1)]

    @staticmethod
    def _get_peak_cut_ind(df, min_diff=0.05):
        df['diff'] = df['Frequency'].diff()
        df[df['diff'] <= min_diff]['diff'] = 0
        max_peak = df['diff'].min()
        if max_peak == 0:
            return df
        return list(df[df['diff'] == max_peak].index)[0]

    def _get_ngram_frequencies_and_indexes(self, data, n, num_samples):
        ngram_info = defaultdict(lambda: {'freq': 0, 'original_indexes': set(), 'new_indexes': set()})
        qualifying_samples = set()

        ngrams = set()
        for item in data:
            for ngram in self._extract_ngrams(item[1], n):
                ngrams.add(ngram)

        for ngram in ngrams:
            for index, item in enumerate(data):
                if ngram in item[1]:
                    ngram_info[ngram]['freq'] += 1
                    ngram_info[ngram]['original_indexes'].add(item[0])
                    ngram_info[ngram]['new_indexes'].add(index)

        for ngram in ngram_info:
            ngram_info[ngram]['original_indexes'] = list(ngram_info[ngram]['original_indexes'])
            ngram_info[ngram]['new_indexes'] = list(ngram_info[ngram]['new_indexes'])
            ngram_info[ngram]['freq'] /= num_samples
            if ngram_info[ngram]['freq'] >= self.min_threshold:
                for index in ngram_info[ngram]['new_indexes']:
                    qualifying_samples.add(index)

        return ngram_info, qualifying_samples

    def _get_frequent_ngrams(self, data):
        n = self.min_ngram
        final_results = {}
        num_samples = len(data)
        while data:
            result, qualifying_samples = self._get_ngram_frequencies_and_indexes(data, n, num_samples)
            result = {ngram: info for ngram, info in result.items() if info['freq'] >= self.min_threshold}
            final_results.update(result)
            if len(qualifying_samples) == 0:
                return final_results
            data = [data[i] for i in qualifying_samples]
            n += 1

    def _filter_longest_substrings_from_equal(self, results):
        ngram_strings = list(results.keys())
        ngram_strings.sort(key=len, reverse=True)
        for long_ngram in ngram_strings:
            if long_ngram not in results:
                continue
            final_key = long_ngram
            final_freq = results[long_ngram]['freq']
            for ngram in ngram_strings:
                if ngram not in results:
                    continue
                if ngram in final_key:
                    if ngram == final_key:
                        continue
                    ngram_freq = results[ngram]['freq']
                    if ngram_freq - final_freq <= self.epsilon:
                        del results[ngram]
                    else:
                        del results[final_key]
                        final_key = ngram
                        final_freq = ngram_freq
        return results

    def _get_significant_cut_ind(self, df):
        significant_df = df[df['Frequency'] >= self.significant_threshold]
        if len(significant_df) > 0:
            return list(significant_df.index)[-1] + 1
        return -1

    def _filter_significant(self, df):
        if len(df) == 1:
            return df
        significant_cut_ind = self._get_significant_cut_ind(df)
        peak_cut_ind = self._get_peak_cut_ind(df)
        return df[:max(significant_cut_ind, peak_cut_ind)].drop(columns=['diff'])

    def run_logic(self, context: Context, dataset_kind):
        """Run check.

        Returns
        -------
        CheckResult
            value is a df with 'Text', 'Number of Samples', 'Frequency' and 'Sample IDs' for each substring.

        Raises
        ------
        DeepchecksValueError
            If the Dataset is empty.
        """
        dataset = context.get_data_by_kind(dataset_kind).sample(self.n_samples, random_state=self.random_state)
        dataset = t.cast(TextData, dataset)
        if dataset.n_samples == 0:
            raise DeepchecksValueError('Dataset cannot be empty')
        data = list(zip(dataset.get_original_text_indexes(), dataset.text))

        substrings_dict = self._get_frequent_ngrams(data)
        substrings_dict = self._filter_longest_substrings_from_equal(substrings_dict)

        if len(substrings_dict) == 0:
            value = {}
            display = None

        else:
            sorted_substrings = sorted(substrings_dict.items(), key=lambda x: (x[1]['freq'], x[0]), reverse=True)
            df = pd.DataFrame({
                'Text': [item[0] for item in sorted_substrings],
                'Frequency': [item[1]['freq'] for item in sorted_substrings],
                'Sample IDs': [item[1]['original_indexes'] for item in sorted_substrings]
            })

            df = self._filter_significant(df)
            df['Number of Samples'] = df['Sample IDs'].str.len()

            value = df
            percent_of_frequent = sum(df['Number of Samples'])
            display = [
                f'{format_percent(percent_of_frequent)} of data samples share common substrings.',
                'Each row in the table shows an example of a frequent substring and the number of times it appears.',
                df[['Text', 'Number of Samples', 'Frequency']].iloc[slice(0, self.n_to_show)]
            ]

        return CheckResult(
            value=value,
            display=display
        )

    def add_condition_zero_result(self, min_substrings: int = 1):
        """Add condition - check that the amount of frequent substrings is below the minimum.

        Parameters
        ----------
        min_substrings : int , default: 1
            minimal amount of frequent substrings allowed.
        """
        def condition(result: Dict) -> ConditionResult:
            num_substrings = len(result)
            msg = f'Found {num_substrings} substrings with ratio above threshold'
            if num_substrings >= min_substrings:
                return ConditionResult(ConditionCategory.WARN, msg)
            else:
                return ConditionResult(ConditionCategory.PASS, msg)

        return self.add_condition(f'There should be not more than {min_substrings} frequent substrings',
                                  condition)
