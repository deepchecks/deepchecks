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
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.nlp import Context, SingleDatasetCheck
from deepchecks.nlp.text_data import TextData
from deepchecks.nlp.utils.text_utils import hash_samples, normalize_samples
from deepchecks.utils.strings import format_list
from deepchecks.utils.other import to_ordional_enumeration
from deepchecks.utils.abstracts.conflicting_labels import ConflictingLabelsAbstract

__all__ = ['ConflictingLabels']


# TODO: docs, text trim
class ConflictingLabels(SingleDatasetCheck, ConflictingLabelsAbstract):
    """Find samples which have the exact same features' values but different labels.

    Parameters
    ----------
    n_to_show : int , default: 5
        number of most common ambiguous samples to show.
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
        self.ignore_case = ignore_case
        self.remove_punctuation = remove_punctuation
        self.normalize_unicode = normalize_unicode
        self.remove_stopwords = remove_stopwords
        self.ignore_whitespace = ignore_whitespace
        self.n_to_show = n_to_show
        self.n_samples = n_samples
        self.random_state = random_state
    
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
        # TODO:
        context.raise_if_multi_label_task(self)
        context.raise_if_token_classification_task(self)
        
        dataset = context.get_data_by_kind(dataset_kind).sample(self.n_samples, random_state=self.random_state)
        dataset = t.cast(TextData, dataset)
        samples = dataset.text
        n_of_samples = len(samples)

        if n_of_samples == 0:
            raise DeepchecksValueError("Dataset cannot be empty")

        samples_hashes = hash_samples(normalize_samples(
            dataset.text, 
            **self._text_normalization_kwargs
        ))
        df = pd.DataFrame({
            "hash": samples_hashes, 
            "Sample ID": dataset.get_original_text_indexes(),
            "Label": dataset.label,
            "Text": dataset.text, 
        })

        by_hash = df.loc[:, ["hash", "Label"]].groupby(["hash"], dropna=False)
        n_of_labels_per_sample = by_hash["Label"].aggregate(lambda x: len(set(x.to_list())))
        ambiguous_samples_hashes = set(n_of_labels_per_sample[n_of_labels_per_sample > 1].index.to_list())

        ambiguous_samples = df[df['hash'].isin(ambiguous_samples_hashes)]
        num_of_ambiguous_samples = ambiguous_samples["Text"].count()
        percent_of_ambiguous_samples = num_of_ambiguous_samples / n_of_samples

        result_df = ambiguous_samples.rename(columns={"hash": "Duplicate"})
        duplicates_enumeration = to_ordional_enumeration(result_df['Duplicate'].to_list())
        result_df["Duplicate"] = result_df["Duplicate"].apply(lambda x: duplicates_enumeration[x])
        result_df = result_df.set_index(["Duplicate", "Sample ID", "Label"])

        result_value = {
            "percent": percent_of_ambiguous_samples,
            "ambiguous_samples": result_df,
        }

        if context.with_display is False:
            return CheckResult(value=result_value)
        
        by_hash = ambiguous_samples.groupby(["hash"], dropna=False)
        fn = lambda x: format_list(x.to_list())
        observed_labels = by_hash["Label"].aggregate(fn)
        instances = by_hash['Sample ID'].aggregate(fn)
        first_in_group = by_hash['Text'].first()

        display_table = (
            pd.DataFrame({
                "Observed Labels": observed_labels,
                "Instances": instances,
                "Text": first_in_group
            })
            .reset_index(drop=True)
            .set_index(["Observed Labels", "Instances"])
        )
        table_description = (
            'Each row in the table shows an example of a data sample '
            'and the its observed labels as found in the dataset.'
        )
        table_note = (
            f'Showing top {self.n_to_show} of {len(display_table)}'
            if self.n_to_show <= len(display_table)
            else ''
        )
        return CheckResult(
            value=result_value, 
            display=[
                table_description,
                table_note,
                # slice over first level of the multiindex
                # in our case it is 'Observed Labels'
                display_table.iloc[slice(0, self.n_to_show)]
            ]
        )

