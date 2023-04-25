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
"""Module contains train-test samples mix check."""
import typing as t
import pandas as pd

from deepchecks.core import CheckResult
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.nlp import Context, TrainTestCheck
from deepchecks.utils.strings import format_percent
from deepchecks.nlp.utils.text_utils import normalize_text, hash_text
from deepchecks.utils.strings import format_list, format_percent
from deepchecks.utils.abstracts.train_test_samples_mix import TrainTestSamplesMixAbstract
from deepchecks.nlp.text_data import TextData
from deepchecks.nlp.checks.data_integrity.text_duplicates import to_ordional_enumeration

__all__ = ['TrainTestSamplesMix']


# TODO: docs
class TrainTestSamplesMix(TrainTestCheck, TrainTestSamplesMixAbstract):
    """Detect samples in the test data that appear also in training data.

    Parameters
    ----------
    n_samples : int , default: 10_000_000
        number of samples to use for this check.
    n_to_show : int , default: 10
        number of samples that appear in test and training data to show.
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
        n_samples: int = 10_000_000,
        n_to_show: int = 10,
        random_state: int = 42,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.ignore_case = ignore_case
        self.remove_punctuation = remove_punctuation
        self.normalize_unicode = normalize_unicode
        self.remove_stopwords = remove_stopwords
        self.ignore_whitespace = ignore_whitespace
        self.n_samples = n_samples
        self.n_to_show = n_to_show
        self.random_state = random_state

    def _prepare_hashes(self, text: t.Iterable[str]) -> t.List[int]:
        return [
            hash_text(normalize_text(
                it,
                ignore_case=self.ignore_case,
                ignore_whitespace=self.ignore_whitespace,
                normalize_uni=self.normalize_unicode,
                remove_punct=self.remove_punctuation,
                remove_stops=self.remove_stopwords,
            ))
            for it in text
        ]

    def run_logic(self, context: Context) -> CheckResult:
        """Run check."""
        train = context.train.sample(self.n_samples, random_state=self.random_state)
        test = context.test.sample(self.n_samples, random_state=self.random_state)
        train = t.cast(TextData, train)
        test = t.cast(TextData, test)
        train_samples = t.cast(t.Sequence[str], train.text)
        test_samples = t.cast(t.Sequence[str], test.text)

        if len(train_samples) == 0:
            raise DeepchecksValueError("Train dataset cannot be empty")
        if len(test_samples) == 0:
            raise DeepchecksValueError("Test dataset cannot be empty")

        train_sample_hashes = self._prepare_hashes(train_samples)
        test_sample_hashes = self._prepare_hashes(test_samples)

        train_df = pd.DataFrame({
            "hash": train_sample_hashes,
            "Text": train_samples,
            "Dataset": ["train" for _ in range(len(train_samples))],
            "Sample ID": train.get_original_text_indexes()
        })
        test_df = pd.DataFrame({
            "hash": test_sample_hashes,
            "Text": test_samples,
            "Dataset": ["test" for _ in range(len(test_samples))],
            "Sample ID": test.get_original_text_indexes()
        })

        hash_intersection = set(train_sample_hashes).intersection(set(test_sample_hashes))
        df = pd.concat([test_df, train_df])
        df = df[df['hash'].isin(hash_intersection)]
        n_of_test_duplicates = df[df['Dataset'] == 'test']['Text'].count()
        n_of_test_samples = test_df.shape[0]
        duplicates_ratio = n_of_test_duplicates / n_of_test_samples

        result_df = df.rename(columns={'hash': 'Duplicate'})
        duplicates_enumeration = to_ordional_enumeration(result_df['Duplicate'].to_list())
        result_df['Duplicate'] = result_df['Duplicate'].apply(lambda x: duplicates_enumeration[x])
        result_df = result_df.set_index(['Duplicate', 'Dataset', 'Sample ID'])
        result_df = result_df.sort_index()

        result_value = {
            "ratio": duplicates_ratio,
            "duplicates": result_df,
        }

        if not (context.with_display and duplicates_ratio > 0):
            return CheckResult(value=result_value)

        train_grouped = df[df['Dataset'] == 'train'].groupby(['hash'], dropna=False)
        train_instances = train_grouped['Sample ID'].aggregate(lambda x: format_list(x.to_list()))

        test_grouped = df[df['Dataset'] == 'test'].groupby(['hash'], dropna=False)
        test_instances = test_grouped['Sample ID'].aggregate(lambda x: format_list(x.to_list()))
        counted_test_duplicates = test_grouped.size()
        first_sample_in_group = test_grouped['Text'].first()

        display_table = pd.DataFrame({
            "Train instances": train_instances,
            "Test instances": test_instances,
            "Test text sample": first_sample_in_group,
            "Number of test duplicates": counted_test_duplicates
        }).reset_index(drop=True).set_index(["Train instances", "Test instances"])

        message = (
            f'{format_percent(duplicates_ratio)} ({n_of_test_duplicates} / {n_of_test_samples}) '
            'of test data samples appear in train data'
        )
        return CheckResult(
            value=result_value,
            display=[message, display_table]
        )
