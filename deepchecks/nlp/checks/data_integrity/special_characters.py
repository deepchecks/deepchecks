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
"""Module contains Special Characters check."""
import random
import string
import typing as t
from collections import Counter

import numpy as np
import pandas as pd
from typing_extensions import Self, TypedDict

from deepchecks.core import CheckResult, ConditionCategory, ConditionResult
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.nlp import Context, SingleDatasetCheck
from deepchecks.nlp._shared_docs import docstrings
from deepchecks.nlp.text_data import TextData
from deepchecks.utils.dataframes import hide_index_for_display
from deepchecks.utils.strings import SPECIAL_CHARACTERS, format_percent

__all__ = ['SpecialCharacters']


class SpecialCharacterInfo(TypedDict):
    samples_ids: t.List[t.Any]
    text_example: str
    percent_of_samples: float


@docstrings
class SpecialCharacters(SingleDatasetCheck):
    """Find samples that contain special characters and also the most common special characters in the dataset.

    Parameters
    ----------
    special_characters_allow_list: Union[str, Sequence[str]] , default ' ' + string.punctuation
        set of special characters to ignore. Punctuation (string.punctuation) is whitelisted by default.
    max_samples_to_show : int , default: 5
        Maximum number of most common special-only samples to show in the display.
    max_special_chars_to_show : int , default: 5
        Maximum number of most common special characters per sample to show in the display.
    max_chars_to_review_per_sample : int , default: 10000
        Maximal number of characters to sample randomly from each text sample.
    n_samples: int, default: 10_000_000
        number of samples to use for this check.
    random_state : int, default: 42
        random seed for all check internals.
    {max_text_length_for_display_param:1*indent}
    """

    SPECIAL_CHARACTERS = frozenset(SPECIAL_CHARACTERS)
    DEFAULT_ALLOW_LIST = frozenset(' ' + string.punctuation)

    def __init__(
            self,
            special_characters_allow_list: t.Union[str, t.Sequence[str], None] = None,
            max_samples_to_show: int = 5,
            max_special_chars_to_show: int = 5,
            max_chars_to_review_per_sample: int = 10000,
            n_samples: int = 1_000_000,
            random_state: int = 42,
            max_text_length_for_display: int = 100,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.special_characters_allow_list = (
            frozenset(special_characters_allow_list)
            if special_characters_allow_list
            else self.DEFAULT_ALLOW_LIST
        )
        self.special_characters_deny_list = self.SPECIAL_CHARACTERS.difference(
            self.special_characters_allow_list
        )
        self.max_samples_to_show = max_samples_to_show
        self.max_special_chars_to_show = max_special_chars_to_show
        self.max_chars_to_review_per_sample = max_chars_to_review_per_sample
        self.n_samples = n_samples
        self.random_state = random_state
        self.max_text_length_for_display = max_text_length_for_display

    def run_logic(self, context: Context, dataset_kind) -> CheckResult:
        """Run check."""
        dataset = context.get_data_by_kind(dataset_kind).sample(self.n_samples, random_state=self.random_state)
        dataset = t.cast(TextData, dataset)

        if dataset.n_samples == 0:
            raise DeepchecksValueError('Dataset cannot be empty')

        samples_per_special_char = {}
        percent_special_chars_in_sample = {}

        for idx, sample in zip(dataset.get_original_text_indexes(), dataset.text):
            if pd.isna(sample):
                continue
            if len(sample) > self.max_chars_to_review_per_sample:
                sample = random.sample(sample, self.max_chars_to_review_per_sample)
            if len(sample) == 0:
                percent_special_chars_in_sample[idx] = 0
                continue
            special_chars_in_sample = [char for char in sample if char in self.special_characters_deny_list]
            percent_special_chars_in_sample[idx] = len(special_chars_in_sample) / len(sample)
            for char in frozenset(special_chars_in_sample):
                base = samples_per_special_char[char] if char in samples_per_special_char else []
                samples_per_special_char[char] = base + [idx]

        percents_arr = np.asarray(list(percent_special_chars_in_sample.values()))
        percent_of_samples_with_special_chars = len(percents_arr[percents_arr > 0]) / dataset.n_samples
        percent_special_chars_in_sample = pd.Series(percent_special_chars_in_sample).sort_values(ascending=False)
        samples_per_special_char = dict(sorted(samples_per_special_char.items(), key=lambda x: -len(x[1])))
        result_value = {
            'samples_per_special_char': samples_per_special_char,
            'percent_of_samples_with_special_chars': percent_of_samples_with_special_chars,
            'percent_special_chars_per_sample': pd.Series(percent_special_chars_in_sample),
        }

        if context.with_display is False or len(samples_per_special_char) == 0:
            return CheckResult(value=result_value)

        display_table = pd.DataFrame(columns=['Sample ID', '% of Special Characters',
                                              'Special Characters', 'Text'])
        for idx, value in percent_special_chars_in_sample[:self.max_samples_to_show].items():
            text_sample = dataset.get_sample_at_original_index(idx)
            special_chars = Counter(char for char in text_sample if char in self.special_characters_deny_list)
            special_chars = [x[0] for x in special_chars.most_common()[:self.max_special_chars_to_show]]
            display_table.loc[len(display_table)] = \
                [idx, value, special_chars, text_sample[:self.max_text_length_for_display]]

        return CheckResult(
            value=result_value,
            display=[
                f'<b>{format_percent(percent_of_samples_with_special_chars)}</b> of samples contain special characters',
                f'List of ignored special characters: {list(self.special_characters_allow_list)}',
                hide_index_for_display(display_table)
            ]
        )

    def add_condition_samples_ratio_w_special_characters_less_or_equal(self: Self, max_ratio: float = 0.05,
                                                                       threshold_ratio_per_sample=0.2) -> Self:
        """Add condition - ratio of samples containing more special characters than threshold is below max_ratio.

        Parameters
        ----------
        max_ratio : float , default: 0.05
            Maximum ratio of samples allowed.
        threshold_ratio_per_sample : float , default: 0.2
            Threshold ratio of special characters in a sample.
        """
        name = f'Ratio of samples containing more than {format_percent(threshold_ratio_per_sample)} ' \
               f'special characters is below {format_percent(max_ratio)}'

        def condition(result: t.Dict[str, t.Any]):
            percents_arr = np.asarray(result['percent_special_chars_per_sample'])
            n_samples_above_threshold = len(percents_arr[percents_arr > threshold_ratio_per_sample])
            if n_samples_above_threshold / len(percents_arr) > max_ratio:
                return ConditionResult(
                    ConditionCategory.FAIL,
                    f'Found {n_samples_above_threshold} samples with special char ratio above threshold'
                )
            else:
                return ConditionResult(
                    ConditionCategory.PASS,
                    f'Found {n_samples_above_threshold} samples with special char ratio above threshold'
                )

        return self.add_condition(name, condition)
