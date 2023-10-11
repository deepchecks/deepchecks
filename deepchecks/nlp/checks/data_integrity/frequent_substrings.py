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
"""Module containing the Frequent Substrings check."""
import re
from collections import defaultdict
from typing import Dict

import pandas as pd

from deepchecks.core import CheckResult, ConditionCategory, ConditionResult
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.nlp import Context, SingleDatasetCheck
from deepchecks.utils.strings import format_percent

__all__ = ['FrequentSubstrings']


class FrequentSubstrings(SingleDatasetCheck):
    """Checks for frequent substrings in the dataset.

    Substrings of varying lengths (n-grams) are extracted from the dataset text samples.
    The frequencies of these n-grams are calculated and only substrings exceeding a defined minimum length are retained.
    The substrings are then sorted by their frequencies and the most frequent substrings are identified.
    Finally, the substrings with the highest frequency and those surpassing a significance level are displayed.

    Parameters
    ----------
    n_to_show : int, default: 5
        Number of most frequent substrings to show.
    n_samples : int, default: 10_000
        Number of samples to use for this check.
    random_state : int, default: 42
        Random seed for all check internals.
    n_sentences : int, default: 5
        The number of sentences to extract from the beginning and end of the text content.
    min_ngram_length: int, default: 4
        Minimum amount of words for a substring to be considered a frequent substring.
    min_substring_ratio: float, default: 0.05
        Minimum frequency required for a substring to be considered "frequent".
    significant_substring_ratio: float, default: 0.3
        Frequency above which samples are considered significant. Substrings
        meeting or exceeding this ratio will always be returned, regardless of
        other parameters and conditions.
    frequency_margin: float, default: 0.02
        Defines the tolerance level for selecting longer overlapping substrings. If a longer substring
        has a frequency that's less than a shorter overlapping substring but the difference is within
        the specified frequency_margin, the longer substring is still preferred.
    min_relative_change : float, optional, default=0.05
        Defines the threshold for relative change. If the computed relative
        change falls below this specified threshold, it is
        considered insignificant and is thus set to zero.

    """

    def __init__(
        self,
        n_to_show: int = 5,
        n_samples: int = 10_000,
        random_state: int = 42,
        n_sentences: int = 5,
        min_ngram_length: int = 4,
        min_substring_ratio: float = 0.05,
        significant_substring_ratio: float = 0.3,
        frequency_margin: float = 0.02,
        min_relative_change: float = 0.05,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_to_show = n_to_show
        self.n_samples = n_samples
        self.random_state = random_state
        self.n_sentences = n_sentences
        self.min_ngram_length = min_ngram_length
        self.min_substring_ratio = min_substring_ratio
        self.significant_substring_ratio = significant_substring_ratio
        self.frequency_margin = frequency_margin
        self.min_relative_change = min_relative_change

    @staticmethod
    def _get_ngrams(text, n):
        """
        Extract n-grams from a given text.

        Parameters:
        -----------
        text : str
            Text from which n-grams are extracted.
        n : int
            Length of the n-grams.

        Returns:
        --------
        List of n-grams.
        """
        if not isinstance(text, str):
            return []
        words = text.split()
        chars = r'(?<=,[.!?]\/)'
        ngrams = []
        for i in range(len(words) - n + 1):
            flag = True
            ngram = words[i:i + n]
            for char in chars:
                if char in ngram:
                    flag = False
                    break
            if flag:
                ngrams.append(' '.join(ngram))
        return ngrams

    @staticmethod
    def _split_sentences(text):
        """
        Split a given text into sentences.

        Args:
            text (str): The input text to be split into sentences.

        Returns:
            list of str: A list of sentences extracted from the input text.
        """
        if not isinstance(text, str):
            return []
        return re.split(r'(?<=[.!?])\s+', text)

    def _get_n_sentences(self, data):
        """
        Extract a specified number of sentences from each item in the input data.

        This function processes each item in the input data, splitting its text content into sentences,
        and then selects a certain number of sentences from the beginning and end of the content.

        Args:
            data (list of tuple): The input data, where each tuple contains item information.

        Returns:
            list of tuple: Processed data with selected sentences for each item.
        """
        for index, item in enumerate(data):
            sentences = self._split_sentences(item[1])
            if len(sentences) > self.n_sentences*2:
                sentences = sentences[:self.n_sentences] + sentences[-self.n_sentences:]
            data[index] = (item[0], ' '.join(sentences))
        return data

    def _calculate_ngram_frequencies(self, data, n, num_samples):
        """
        Calculate the frequencies of n-grams in the provided data.

        For each n-gram extracted from the dataset, the method computes its frequency
        and keeps track of the original and filtered indexes where the n-gram occurs.
        Only n-grams that have a frequency greater than or equal to `self.min_substring_ratio`
        are retained in the results.

        Parameters:
        -----------
        data : list of tuple
            The dataset from which to extract n-grams. Each tuple consists of
            an original index and a text string.
        n : int
            The length of the n-grams to be extracted.
        num_samples : int
            The total number of samples in the dataset.

        Returns:
        --------
        tuple
            A tuple containing two items:
            1. A dictionary where keys are n-grams and values are another dictionary
               containing frequency ('freq') and original indexes ('original_indexes') of the n-gram.
            2. A set containing indexes of filtered samples that have qualifying n-grams.

        Notes:
        ------
        The method uses `self._get_ngrams` to extract n-grams from text strings
        and `self.min_substring_ratio` as the threshold for deciding which n-grams
        are frequent enough to be included in the results.
        """
        ngram_info = defaultdict(lambda: {'freq': 0, 'original_indexes': [], 'filtered_indexes': []})
        filtered_samples = set()

        for index, item in enumerate(data):
            for ngram in self._get_ngrams(item[1], n):
                ngram_info[ngram]['original_indexes'].append(item[0])
                ngram_info[ngram]['filtered_indexes'].append(index)

        ngrams = list(ngram_info.keys())
        for ngram in ngrams:
            ngram_freq = len(ngram_info[ngram]['original_indexes'])/num_samples
            if ngram_freq >= self.min_substring_ratio:
                ngram_info[ngram]['freq'] = ngram_freq
                for index in ngram_info[ngram]['filtered_indexes']:
                    filtered_samples.add(index)
                del ngram_info[ngram]['filtered_indexes']
            else:
                del ngram_info[ngram]

        return ngram_info, filtered_samples

    def _find_frequent_substrings(self, data):
        """
        Identify and return the frequent substrings (n-grams) from the provided data.

        Starting from the n-grams of length `self.min_ngram_length`, the method extracts
        and computes the frequencies of n-grams iteratively. For each iteration,
        it filters the data to only include samples that contain the frequent n-grams
        identified in that iteration. The process continues by increasing the n-gram length
        until no frequent n-grams are identified in an iteration.

        Parameters:
        -----------
        data : list of tuple
            The dataset from which to extract n-grams. Each tuple consists of
            an original index and a text string.

        Returns:
        --------
        dict
            A dictionary where keys are frequent n-grams and values are another dictionary
            containing frequency ('freq') and original indexes ('original_indexes') of the n-gram.

        Notes:
        ------
        The method relies on `self._calculate_ngram_frequencies` to compute the
        frequencies of n-grams and identify the frequent ones.
        """
        n = self.min_ngram_length
        final_results = {}
        num_samples = len(data)
        while data:
            ngram_info, filtered_samples = self._calculate_ngram_frequencies(data, n, num_samples)
            final_results.update(ngram_info)
            if len(filtered_samples) == 0:
                return final_results
            data = [data[i] for i in filtered_samples]
            n += 1

    def _eliminate_overlapping_substrings(self, results):
        """
        Remove overlapping n-grams from the results based on their lengths and frequencies.

        Given a dictionary of n-grams and their respective frequencies, this method
        filters out overlapping n-grams, preserving only the longest n-gram, unless
        a shorter n-gram has a frequency that exceeds the longer n-gram's frequency
        by a margin greater than `self.frequency_margin`.

        Parameters:
        -----------
        results : dict
            A dictionary where keys are n-grams and values are another dictionary
            containing frequency ('freq'), original indexes ('original_indexes'),
            and filtered indexes ('filtered_indexes') of the n-gram.

        Returns:
        --------
        dict
            A filtered dictionary where keys are non-overlapping n-grams and values are
            details about the n-grams similar to the input dictionary.

        Notes:
        ------
        The method employs a nested loop approach, comparing each n-gram with every other
        n-gram to identify and eliminate overlapping n-grams based on length and frequency criteria.
        """
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
                    if ngram_freq - final_freq <= self.frequency_margin:
                        del results[ngram]
                    else:
                        del results[final_key]
                        final_key = ngram
                        final_freq = ngram_freq
        return results

    def _get_significant_cut_ind(self, df):
        """
        Determine the index cutoff for substrings with frequencies above a significant threshold.

        This method identifies the position in the dataframe 'df' where the substring's frequency
        surpasses the defined threshold `self.significant_substring_ratio`. If there are multiple
        substrings meeting the criteria, it returns the index after the last such substring. If no
        such substring exists, it returns -1.

        Parameters:
        -----------
        df : pd.DataFrame
            A sorted Dataframe containing substring information. Expected to have a 'Frequency' column
            that lists the frequency of each substring in the dataset.

        Returns:
        --------
        int
            The index after the last substring that meets the significant frequency threshold or
            -1 if no such substring exists.

        Notes:
        ------
        The method is useful for filtering out substrings below a certain significance level based
        on their frequencies.
        """
        significant_df = df[df['Frequency'] >= self.significant_substring_ratio]
        if len(significant_df) > 0:
            return list(significant_df.index)[-1] + 1
        return -1

    def _identify_peak_cut(self, df):
        """
        Identify the index at which there's a maximum relative change (peak) in the 'Frequency' column.

        The function calculates the absolute difference between consecutive values in the 'Frequency' column
        and divides this by the previous row's value to find the relative change or ratio. If the relative
        change is below a certain threshold (`self.min_relative_change`), it is set to zero. The index of the maximum
        relative change (peak) is returned.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with a 'Frequency' column.

        Returns
        -------
        int
            The index of the maximum relative change in the 'Frequency' column. If no significant peak is
            found, returns the length of the dataframe.
        """
        diff = df['Frequency'].diff().abs()
        ratio = diff.div(df['Frequency'].shift(1))
        ratio[ratio <= self.min_relative_change] = 0
        max_peak = ratio.max()
        if max_peak == 0:
            return len(df)
        return list(ratio[ratio == max_peak].index)[0]

    def _filter_peak_and_significant(self, df):
        """
        Filter the dataframe to retain only the significant substrings.

        The function first identifies the cut-off index based on the significance ratio. Then, it identifies
        a cut-off based on the peak in the frequency ratio. The final dataframe is truncated using the maximum
        of these two cut-off indices.

        Parameters
        ----------
        df : DataFrame
            The input dataframe expected to have a 'Frequency' column.

        Returns
        -------
        DataFrame
            A truncated dataframe containing only the significant substrings based on the established criteria.
        """
        if len(df) == 1:
            return df
        significant_cut_ind = self._get_significant_cut_ind(df)
        peak_cut_ind = self._identify_peak_cut(df)
        return df[:max(significant_cut_ind, peak_cut_ind)]

    def run_logic(self, context: Context, dataset_kind):
        """Run check.

        Parameters:
        -----------
        context : Context
            Contains dataset and related methods.
        dataset_type :
            Type or format of the dataset.

        Returns
        -------
        CheckResult
            Results containing frequent substrings' information.

        Raises
        ------
        DeepchecksValueError
            If the Dataset is empty.
        """
        dataset = context.get_data_by_kind(dataset_kind)
        dataset_sampled = dataset.sample(self.n_samples, random_state=self.random_state)
        if dataset_sampled.n_samples == 0:
            raise DeepchecksValueError('Dataset cannot be empty')
        data = list(zip(dataset_sampled.get_original_text_indexes(), dataset_sampled.text))

        data = self._get_n_sentences(data)

        substrings_dict = self._find_frequent_substrings(data)
        substrings_dict = self._eliminate_overlapping_substrings(substrings_dict)

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

            df = self._filter_peak_and_significant(df)

            if self.n_samples < len(dataset):
                for substring_index, substring in enumerate(df['Text']):
                    indexes = []
                    for sample_index, sample in enumerate(dataset):
                        if substring in sample:
                            indexes.append(sample_index)
                    df.at[substring_index, 'Sample IDs'] = indexes
                    df.at[substring_index, 'Frequency'] = len(indexes) / len(dataset)
                df = df.sort_values(by=['Frequency', 'Text'], ascending=False)

            df['Number of Samples'] = df['Sample IDs'].str.len()
            df['% In data'] = df['Frequency'].apply(format_percent)

            value = df.to_dict()
            percent_of_frequent = len(set(sum(df['Sample IDs'], [])))/len(dataset)
            if context.with_display:
                display = [
                    f'{format_percent(percent_of_frequent)} of data samples share common substrings.',
                    'Each row in the table shows an example of a frequent substring '
                    'and the number of times it appears.',
                    df[['Text', 'Number of Samples', '% In data']].iloc[slice(0, self.n_to_show)]
                ]
            else:
                display = None
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
            if len(result) == 0:
                num_substrings = 0
            else:
                num_substrings = len(result['Text'])
            msg = f'Found {num_substrings} substrings with ratio above threshold'
            if num_substrings >= min_substrings:
                return ConditionResult(ConditionCategory.WARN, msg)
            else:
                return ConditionResult(ConditionCategory.PASS, msg)

        return self.add_condition(f'No more than {min_substrings} substrings with ratio above '
                                  f'{self.min_substring_ratio}', condition)
