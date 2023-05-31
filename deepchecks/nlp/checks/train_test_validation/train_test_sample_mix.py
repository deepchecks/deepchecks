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

import numpy as np
import pandas as pd

from deepchecks.core import CheckResult
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.nlp import Context, TrainTestCheck
from deepchecks.nlp._shared_docs import docstrings
from deepchecks.nlp.text_data import TextData
from deepchecks.nlp.utils.text import cut_string, hash_samples, normalize_samples
from deepchecks.utils.abstracts.train_test_samples_mix import TrainTestSamplesMixAbstract
from deepchecks.utils.other import to_ordional_enumeration
from deepchecks.utils.strings import format_list, format_percent, truncate_string

__all__ = ['TrainTestSamplesMix']


@docstrings
class TrainTestSamplesMix(TrainTestCheck, TrainTestSamplesMixAbstract):
    """Detect samples in the test data that appear also in training data.

    Parameters
    ----------
    {text_normalization_params:1*indent}
    n_samples : int , default: 10_000_000
        number of samples to use for this check.
    n_to_show : int , default: 10
        number of samples that appear in test and training data to show.
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
            n_samples: int = 10_000_000,
            n_to_show: int = 10,
            random_state: int = 42,
            max_text_length_for_display: int = 30,
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

    def _get_duplicate_indices(self, train: TextData, test: TextData,
                               train_samples: t.Sequence[str], test_samples: t.Sequence[str]):
        normalization_kwargs = self._text_normalization_kwargs
        train_sample_hashes = hash_samples(normalize_samples(train_samples, **normalization_kwargs))
        test_sample_hashes = hash_samples(normalize_samples(test_samples, **normalization_kwargs))

        train_df = pd.DataFrame({
            'hash': train_sample_hashes,
            'Text': train_samples,
            'Dataset': ['train' for _ in range(len(train_samples))],
            'Sample ID': train.get_original_text_indexes()
        })
        test_df = pd.DataFrame({
            'hash': test_sample_hashes,
            'Text': test_samples,
            'Dataset': ['test' for _ in range(len(test_samples))],
            'Sample ID': test.get_original_text_indexes()
        })

        hash_intersection = set(train_sample_hashes).intersection(set(test_sample_hashes))
        df = pd.concat([test_df, train_df])
        return df['hash'].isin(hash_intersection), df

    def run_logic(self, context: Context) -> CheckResult:
        """Run check."""
        train = context.train.sample(self.n_samples, random_state=self.random_state)
        test = context.test.sample(self.n_samples, random_state=self.random_state)
        train = t.cast(TextData, train)
        test = t.cast(TextData, test)
        train_samples = t.cast(t.Sequence[str], train.text)
        test_samples = t.cast(t.Sequence[str], test.text)
        n_of_test_samples = len(test_samples)

        if len(train_samples) == 0:
            raise DeepchecksValueError('Train dataset cannot be empty')
        if len(test_samples) == 0:
            raise DeepchecksValueError('Test dataset cannot be empty')

        # First run on truncated dataset
        train_truncated = [cut_string(x) for x in train_samples]
        test_truncated = [cut_string(x) for x in test_samples]

        duplicate_bool_df = self._get_duplicate_indices(train, test, train_truncated, test_truncated)[0]
        train_indices_reinspect = duplicate_bool_df.iloc[len(test_samples):]
        train_indices_reinspect = np.where(train_indices_reinspect.values)[0]
        test_indices_reinspect = duplicate_bool_df.iloc[:len(test_samples)]
        test_indices_reinspect = np.where(test_indices_reinspect.values)[0]

        # keep only samples that where found to be duplicates after cut_string
        train = train.copy(train_indices_reinspect.tolist())
        test = test.copy(test_indices_reinspect.tolist())

        train_samples = t.cast(t.Sequence[str], train.text)
        test_samples = t.cast(t.Sequence[str], test.text)

        if (len(train_samples) == 0) or (len(test_samples) == 0):
            result_value = {
                'ratio': 0,
                'duplicates': pd.DataFrame(
                    index=pd.MultiIndex(levels=[[], [], []], codes=[[], [], []],
                                        names=['Duplicate', 'Dataset', 'Sample ID']),
                    columns=['Text'])
            }
            return CheckResult(value=result_value)

        bool_df, df = self._get_duplicate_indices(train, test, train_samples, test_samples)
        df = df[bool_df]

        n_of_test_duplicates = df[df['Dataset'] == 'test']['Text'].count()
        duplicates_ratio = n_of_test_duplicates / n_of_test_samples

        result_df = df.rename(columns={'hash': 'Duplicate'})
        duplicates_enumeration = to_ordional_enumeration(result_df['Duplicate'].to_list())
        result_df['Duplicate'] = result_df['Duplicate'].apply(lambda x: duplicates_enumeration[x])
        result_df = result_df.set_index(['Duplicate', 'Dataset', 'Sample ID'])
        result_df = result_df.sort_index()

        result_value = {
            'ratio': duplicates_ratio,
            'duplicates': result_df
        }

        if context.with_display is False or duplicates_ratio == 0:
            return CheckResult(value=result_value)

        train_grouped = df[df['Dataset'] == 'train'].groupby(['hash'], dropna=False)
        train_instances = train_grouped['Sample ID'].aggregate(lambda x: format_list(x.to_list()))

        test_grouped = df[df['Dataset'] == 'test'].groupby(['hash'], dropna=False)
        test_instances = test_grouped['Sample ID'].aggregate(lambda x: format_list(x.to_list()))
        counted_test_duplicates = test_grouped.size()
        first_sample_in_group = test_grouped['Text'].first()

        display_table = pd.DataFrame({
            'Train Sample IDs': train_instances,
            'Test Sample IDs': test_instances,
            'Test Text Sample': first_sample_in_group.apply(self._truncate_text),
            'Number of Test Duplicates': counted_test_duplicates
        })

        display_table = display_table.iloc[:self.n_to_show]
        display_table = display_table.reset_index(drop=True).set_index(['Train Sample IDs', 'Test Sample IDs'])

        message = (
            f'{format_percent(duplicates_ratio)} ({n_of_test_duplicates} / {n_of_test_samples}) '
            'of test data samples also appear in train data'
        )
        return CheckResult(
            value=result_value,
            display=[message, display_table]
        )
