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
import string
import typing as t

import pandas as pd
from typing_extensions import Self, TypedDict

from deepchecks.core import CheckResult, ConditionCategory, ConditionResult
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.nlp import Context, SingleDatasetCheck
from deepchecks.nlp._shared_docs import docstrings
from deepchecks.nlp.text_data import TextData
from deepchecks.utils.strings import SPECIAL_CHARACTERS, format_list, format_percent
from deepchecks.utils.strings import get_ellipsis as truncate_string

__all__ = ['SpecialCharacters']


class SpecialCharacterInfo(TypedDict):
    samples_ids: t.List[t.Any]
    text_example: str
    percent_of_samples: float


class ResultValue(TypedDict):
    special_characters: t.Dict[str, SpecialCharacterInfo]
    total_percent_of_samples_with_spec_chars: float


@docstrings
class SpecialCharacters(SingleDatasetCheck):
    """Find samples that contain special characters.

    Parameters
    ----------
    special_characters_whitelist: Union[str, Sequence[str]] , default ' ' + string.punctuation
        set of special characters to ignore. Punctuation (string.punctuation) is whitelisted by default.
    n_most_common : int , default: 10
        Number of most common special-only samples to show in results
    n_samples: int, default: 10_000_000
        number of samples to use for this check.
    random_state : int, default: 42
        random seed for all check internals.
    {max_text_length_for_display_param:1*indent}
    """

    SPECIAL_CHARACTERS = frozenset(SPECIAL_CHARACTERS)
    DEFAULT_WHILTELIST = frozenset(' ' + string.punctuation)

    def __init__(
        self,
        special_characters_whitelist: t.Union[str, t.Sequence[str], None] = None,
        n_most_common: int = 10,
        n_samples: int = 10_000_000,
        random_state: int = 42,
        max_text_length_for_display: int = 30,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.special_characters_whitelist = (
            frozenset(special_characters_whitelist)
            if special_characters_whitelist
            else self.DEFAULT_WHILTELIST
        )
        self.special_characters = self.SPECIAL_CHARACTERS.difference(
            self.special_characters_whitelist
        )
        self.n_most_common = n_most_common
        self.n_samples = n_samples
        self.random_state = random_state
        self.max_text_length_for_display = max_text_length_for_display

    def run_logic(self, context: Context, dataset_kind) -> CheckResult:
        """Run check."""
        dataset = context.get_data_by_kind(dataset_kind).sample(self.n_samples, random_state=self.random_state)
        dataset = t.cast(TextData, dataset)
        samples = dataset.text
        n_of_samples = len(samples)

        if n_of_samples == 0:
            raise DeepchecksValueError('Dataset cannot be empty')

        special_characters = self.special_characters
        data: t.Dict[str, SpecialCharacterInfo] = {}
        samples_with_spec_chars = set()

        for idx, sample in zip(dataset.get_original_text_indexes(), samples):
            intersection = frozenset(sample).intersection(special_characters)
            if intersection:
                samples_with_spec_chars.add(idx)
                for char in intersection:
                    data[char] = data.get(char, {
                        'samples_ids': [],
                        'text_example': sample,
                        'percent_of_samples': 0
                    })
                    data[char]['samples_ids'].append(idx)

        for char, info in data.items():
            info['percent_of_samples'] = len(info['samples_ids']) / n_of_samples

        result_value = ResultValue(
            special_characters=data,
            total_percent_of_samples_with_spec_chars=len(samples_with_spec_chars) / n_of_samples
        )

        if context.with_display is False or len(data) == 0:
            return CheckResult(value=result_value)

        display_table = pd.DataFrame(
            index=range(len(data)),
            columns=['Special Character', '% of Samples With Character', 'Sample IDs', 'Text Example'],
            data=[
                [char,
                 values['percent_of_samples'],
                 format_list(values['samples_ids']),
                 truncate_string(values['text_example'], self.max_text_length_for_display)]
                for char, values in data.items()
            ],
        )
        display_table = (
            display_table.sort_values(by=['% of Samples With Character'], ascending=False)
            .reset_index(drop=True)
            .set_index(['Special Character'])
        )
        if self.n_most_common > display_table.shape[0]:
            message = ''
        else:
            message = (
                f'Showing only the top {self.n_most_common} most common special characters, '
                'you can change it using n_most_common param.'
            )
        return CheckResult(
            value=result_value,
            display=[
                f'List of ignored special characters: {list(self.special_characters_whitelist)}',
                message,
                display_table.iloc[:self.n_most_common]
            ]
        )

    def add_condition_ratio_of_special_characters_less_or_equal(self: Self, max_ratio: float = 0.05) -> Self:
        """Add condition - each special character ratio is less or equal to the threshold.

        Parameters
        ----------
        max_ratio : float , default: 0.05
            Maximum ratio allowed.
        """
        name = f'Ratio of each special character is less or equal to {format_percent(max_ratio)}'

        def condition(result: ResultValue):
            not_passed = {
                k: format_percent(v['percent_of_samples'])
                for k, v in result['special_characters'].items()
                if v['percent_of_samples'] > max_ratio
            }
            if not_passed:
                return ConditionResult(
                    ConditionCategory.WARN,
                    f'Found {len(not_passed)} special characters with ratio above threshold: {not_passed}'
                )
            return ConditionResult(
                ConditionCategory.PASS,
                'No special characters with ratio above threshold found'
            )

        return self.add_condition(name, condition)

    def add_condition_ratio_of_samples_with_special_characters_less_or_equal(
        self: Self,
        max_ratio: float = 0.05
    ) -> Self:
        """Add condition - ratio of samples with special character is less or equal to the threshold.

        Parameters
        ----------
        max_ratio : float , default: 0.05
            Maximum ratio allowed.
        """
        name = f'Ratio of samples with special character is less or equal to {format_percent(max_ratio)}'

        def condition(result: ResultValue):
            ratio = result['total_percent_of_samples_with_spec_chars']
            details = f'Ratio of samples with special characters is {format_percent(ratio)}'
            category = ConditionCategory.WARN if ratio > max_ratio else ConditionCategory.PASS
            return ConditionResult(category, details)

        return self.add_condition(name, condition)
