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
"""module contains Data Duplicates check."""
import typing as t

import pandas as pd

from deepchecks.core import CheckResult
from deepchecks.nlp import Context, SingleDatasetCheck
from deepchecks.utils.abstracts.data_duplicates import DataDuplicatesAbstract
from deepchecks.nlp.utils.text_utils import normalize_text, hash_text
from deepchecks.nlp.text_data import TextData
from deepchecks.utils.strings import format_list, format_percent
from deepchecks.utils.other import to_ordional_enumeration

__all__ = ['TextDuplicates']


# TODO: docs
class TextDuplicates(SingleDatasetCheck, DataDuplicatesAbstract):
    """Checks for duplicate samples in the dataset.

    Parameters
    ----------
    n_to_show : int , default: 5
        number of most common duplicated samples to show.
    n_samples : int , default: 10_000_000
        number of samples to use for this check.
    random_state : int, default: 42
        random seed for all check internals.
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

    def run_logic(self, context: Context, dataset_kind):
        """Run check."""
        dataset = context.get_data_by_kind(dataset_kind).sample(self.n_samples, random_state=self.random_state)
        dataset = t.cast(TextData, dataset)
        samples = dataset.text

        sample_hashes = [
            hash_text(normalize_text(
                it,
                ignore_case=self.ignore_case,
                ignore_whitespace=self.ignore_whitespace,
                normalize_uni=self.normalize_unicode,
                remove_punct=self.remove_punctuation,
                remove_stops=self.remove_stopwords,
            ))
            for it in samples
        ]

        df = pd.DataFrame({
            "Text": samples,
            "hash": sample_hashes,
            "Sample ID": dataset.get_original_text_indexes()
        })
        grouped_samples = df.groupby(by=["hash"], dropna=False)
        counted_samples = grouped_samples['Text'].size()
        n_of_unique = len(counted_samples)
        n_of_samples = df.shape[0]
        percent_of_duplicates = 1 - (1.0 * n_of_unique) / (1.0 * n_of_samples)

        counted_duplicates = counted_samples[counted_samples > 1]
        duplicates_hashes = set(counted_duplicates.index)
        value = df[df['hash'].isin(duplicates_hashes)].sort_values(by=["hash"])
        value = value.rename(columns={"hash": "Duplicate"})
        duplicates_enumeration = to_ordional_enumeration(value['Duplicate'].to_list())

        value['Duplicate'] = value['Duplicate'].apply(lambda x: duplicates_enumeration[x])
        value = value.set_index(["Duplicate", "Sample ID"])

        result_value = {
            "percent_of_duplicates": percent_of_duplicates,
            "duplicates": value
        }

        if not (context.with_display and percent_of_duplicates > 0):
            return CheckResult(value=result_value)

        first_sample = grouped_samples['Text'].first()
        instances = grouped_samples['Sample ID'].aggregate(lambda x: format_list(x.to_list()))

        # TODO: refactor
        table = pd.DataFrame({
            "Text": first_sample,
            "Instances": instances,
            "Number of Samples": counted_duplicates
        })
        table = table[table["Number of Samples"] > 1]
        table = table.set_index(["Instances", "Number of Samples"])

        return CheckResult(
            value=result_value,
            display=[
                f'{format_percent(percent_of_duplicates)} of data samples are duplicates. ',
               'Each row in the table shows an example of a text duplicate and the number of times it appears.',
               table
            ]
        )