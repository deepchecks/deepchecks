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
"""Module contains SpecialCharacters check."""
import string
import typing as t

import pandas as pd
from typing_extensions import Self

from deepchecks.core import CheckResult, ConditionCategory, ConditionResult
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.nlp import Context, SingleDatasetCheck
from deepchecks.nlp.text_data import TextData
from deepchecks.utils.strings import format_list, format_percent
from deepchecks.utils.strings import get_ellipsis as truncate_string

__all__ = ['SpecialCharacters']


SPECIAL_CHARACTERS = frozenset(string.punctuation)


# TODO: docs
class SpecialCharacters(SingleDatasetCheck):
    """Search in column[s] for values that contains only special characters.

    Parameters
    ----------
    columns : Union[Hashable, List[Hashable]] , default: None
        Columns to check, if none are given checks all columns except ignored ones.
    ignore_columns : Union[Hashable, List[Hashable]] , default: None
        Columns to ignore, if none given checks based on columns variable.
    n_most_common : int , default: 2
        Number of most common special-only samples to show in results
    n_samples: int = 10_000_000,
        random_state: int = 42,
    """

    def __init__(
        self,
        special_characters_whitelist: t.Union[str, t.Sequence[str], None] = None,
        ignore_case: bool = True,
        remove_punctuation: bool = True,
        normalize_unicode: bool = True,
        remove_stopwords: bool = True,
        ignore_whitespace: bool = False,
        n_most_common: int = 10,
        n_samples: int = 10_000_000,
        max_text_length_for_display: int = 30,
        random_state: int = 42,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.special_characters_whitelist = (
            frozenset(special_characters_whitelist)
            if special_characters_whitelist
            else frozenset()
        )
        self.special_characters = SPECIAL_CHARACTERS.difference(self.special_characters_whitelist)
        self.ignore_case = ignore_case
        self.remove_punctuation = remove_punctuation
        self.normalize_unicode = normalize_unicode
        self.remove_stopwords = remove_stopwords
        self.ignore_whitespace = ignore_whitespace
        self.n_most_common = n_most_common
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

    def run_logic(self, context: Context, dataset_kind) -> CheckResult:
        """Run check."""
        dataset = context.get_data_by_kind(dataset_kind).sample(self.n_samples, random_state=self.random_state)
        dataset = t.cast(TextData, dataset)
        samples = dataset.text
        n_of_samples = len(samples)

        if n_of_samples == 0:
            raise DeepchecksValueError("Dataset cannot be empty")

        data = {}

        for char in self.special_characters:
            for idx, sample in zip(dataset.get_original_text_indexes(), samples):
                if char in sample:
                    data[char] = data.get(char, {"samples_ids": [], "text_example": sample})
                    data[char]['samples_ids'].append(idx)

        for char in data.keys():
            data[char]["percent_of_samples"] = len(data[char]['samples_ids']) / n_of_samples

        if context.with_display is False:
            return CheckResult(value=data)

        display_table = pd.DataFrame(
            index=range(len(data)),
            columns=["Special Character", "% of Samples With Character", "Instances", "Text Example"],
            data=[
                [char,
                 values["percent_of_samples"],
                 format_list(values['samples_ids']),
                 truncate_string(values["text_example"], self.max_text_length_for_display)]
                for char, values in data.items()
            ],
        )
        display_table = (
            display_table.sort_values(by=["% of Samples With Character"], ascending=False)
            .reset_index(drop=True)
            .set_index(["Special Character"])
        )
        if self.n_most_common > display_table.shape[0]:
            message = ""
        else:
            message = (
                f'Showing only the top {self.n_most_common} most common characters, '
                'you can change it using n_most_common param'
            )
        return CheckResult(
            value=data,
            display=[message, display_table.iloc[:self.n_most_common]]
        )

    # TODO:
    # is default max_ratio good?
    # are method and condition names good?
    def add_condition_ratio_of_special_characters_less_or_equal(self: Self, max_ratio: float = 0.05) -> Self:
        """Add condition - ratio of special character is less or equal to the threshold.

        Parameters
        ----------
        max_ratio : float , default: 0.05
            Maximum ratio allowed.
        """
        name = f'Ratio of each special character is less or equal to {format_percent(max_ratio)}'

        def condition(result: t.Dict[str, t.Dict[str, t.Any]]):
            not_passed = {
                k: format_percent(v['percent_of_samples'])
                for k, v in result.items()
                if v['percent_of_samples'] > max_ratio
            }
            if not_passed:
                return ConditionResult(
                    ConditionCategory.WARN,
                    f'Found {len(not_passed)} special characters with ratio above threshold: {not_passed}'
                )
            return ConditionResult(
                ConditionCategory.PASS,
                "" # TODO:
            )

        return self.add_condition(name, condition)