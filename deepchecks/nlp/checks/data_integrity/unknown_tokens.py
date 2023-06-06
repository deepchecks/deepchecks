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
"""Module contains the Unknown Tokens check."""
import contextlib
import copy
import sys
import typing as t
import warnings
from collections import Counter

import nltk
import plotly.graph_objects as go

from deepchecks.core import CheckResult, ConditionCategory, ConditionResult
from deepchecks.core.errors import DeepchecksProcessError, DeepchecksValueError
from deepchecks.nlp import Context, SingleDatasetCheck
from deepchecks.nlp._shared_docs import docstrings
from deepchecks.nlp.text_data import TextData
from deepchecks.utils.numbers import round_sig
from deepchecks.utils.strings import format_list, format_percent, truncate_string

__all__ = ['UnknownTokens']

OTHER_CAT_NAME = 'Other Unknown Words'


@docstrings
class UnknownTokens(SingleDatasetCheck):
    """Find samples that contain tokens unsupported by your tokenizer.

    Parameters
    ----------
    tokenizer: t.Any , default: None
        Tokenizer from the HuggingFace transformers library to use for tokenization. If None,
        AutoTokenizer.from_pretrained('bert-base-uncased') will be used.
    group_singleton_words: bool, default: False
        If True, group all words that appear only once in the data into the "Other" category in the display.
    n_most_common : int , default: 5
        Number of most common words with unknown tokens to show in the display.
    n_samples: int, default: 1_000_000
        number of samples to use for this check.
    random_state : int, default: 42
        random seed for all check internals.
    {max_text_length_for_display_param:1*indent}
    """

    def __init__(
        self,
        tokenizer: t.Any = None,
        group_singleton_words: bool = False,
        n_most_common: int = 5,
        n_samples: int = 1_000_000,
        random_state: int = 42,
        max_text_length_for_display: int = 30,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.tokenizer = copy.deepcopy(tokenizer)
        if tokenizer is None:
            try:
                from transformers import AutoTokenizer  # pylint: disable=W0611,C0415 # noqa
            except ImportError as e:
                raise DeepchecksProcessError(
                    'Tokenizer was not provided. In order to use checks default '
                    'tokenizer (bert-base-uncased), please run:\n>> pip install transformers>=4.27.4.'
                ) from e
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        else:
            self._validate_tokenizer()
        self._use_fast_method = self.tokenizer.is_fast
        # We're not feeding the data to a model, so we don't need to limit the length
        self.tokenizer.model_max_length = sys.maxsize

        self.group_singleton_words = group_singleton_words
        self.n_most_common = n_most_common
        self.n_samples = n_samples
        self.random_state = random_state
        self.max_text_length_for_display = max_text_length_for_display

    def _validate_tokenizer(self):
        # TODO: add ability to pass spacy and nltk tokenizers
        if not hasattr(self.tokenizer, 'tokenize'):
            raise DeepchecksValueError('tokenizer must have a "tokenize" method')
        if not hasattr(self.tokenizer, 'unk_token_id'):
            raise DeepchecksValueError('tokenizer must have an "unk_token_id" attribute')
        if not hasattr(self.tokenizer, 'convert_tokens_to_ids'):
            raise DeepchecksValueError('tokenizer must have an "convert_tokens_to_ids" method')
        if not hasattr(self.tokenizer, 'is_fast'):
            raise DeepchecksValueError('tokenizer must have an "is_fast" method')

    def run_logic(self, context: Context, dataset_kind) -> CheckResult:
        """Run check."""
        dataset = context.get_data_by_kind(dataset_kind).sample(self.n_samples, random_state=self.random_state)
        dataset = t.cast(TextData, dataset)
        samples = dataset.text
        if len(samples) == 0:
            raise DeepchecksValueError('Dataset cannot be empty')

        indices = dataset.get_original_text_indexes()

        all_unknown_words_counter, total_words, unknown_word_indexes = self.find_unknown_words(samples, indices)
        if len(all_unknown_words_counter) == 0:
            display = None
            value = {'unknown_word_ratio': 0,
                     'unknown_word_details': {}}
        else:
            fig = self.create_pie_chart(all_unknown_words_counter, total_words)
            percent_explanation = (
                '<p style="font-size:0.9em;line-height:1;"><i>'
                'Percents shown above are the percent of each word (or group of words) out of all words in the data.'
            )
            display = [fig, percent_explanation]

            # The value contains two fields - unknown_word_percent and unknown_word_details.
            # The latter contains a dict, in which for each word we have its ratio of the data and the list of indexes
            # of the samples that contain it.
            unknown_word_details = {}
            for word, indexes in unknown_word_indexes.items():
                unknown_word_details[word] = {'ratio': all_unknown_words_counter[word] / total_words,
                                              'indexes': indexes}
            value = {'unknown_word_ratio': sum(all_unknown_words_counter.values()) / total_words,
                     'unknown_word_details': unknown_word_details}

        return CheckResult(value, display=display)

    def _get_non_text_token_ids(self):
        """Get ids of all non-text tokens in the tokenizer.

        These include notably the [CLS] token marking the beginning of the sequence, the [SEP] token marking the end
        of the sequence, and the [PAD] token used for padding.
        """
        non_text_token_ids = []
        for token_name, token in self.tokenizer.special_tokens_map.items():
            if token_name not in ['unk_token']:
                non_text_token_ids.append(self.tokenizer.convert_tokens_to_ids(token))
        return non_text_token_ids

    def find_unknown_words(self, samples, indices):
        """Find words with unknown tokens in samples."""
        all_unknown_tokens = []
        unknown_token_indexes = {}
        total_tokens = 0

        if self._use_fast_method:
            non_text_token_ids = self._get_non_text_token_ids()

            # Batch tokenization
            # ------------------
            # Needed to avoid warning when used after loading a hub dataset
            # We divert the printing to stdout (done by the rust code within the HuggingFace tokenizer)
            # into this filter, that will filter out any print containing the str 'huggingface/tokenizers'
            # This warning printout is activated when running this check after loading a HuggingFace dataset,
            # and is irrelevant to us because we're not forking the process.
            # see: https://github.com/huggingface/transformers/issues/5486
            with contextlib.redirect_stdout(PrintFilter(sys.stdout)):
                tokenized_samples = self.tokenizer(list(samples), return_offsets_mapping=True,
                                                   is_split_into_words=False, truncation=False)

            for idx, (tokens, offsets_mapping, sample) in zip(indices, zip(tokenized_samples['input_ids'],
                                                                           tokenized_samples['offset_mapping'],
                                                                           samples)):
                for token_id, offset_mapping in zip(tokens, offsets_mapping):
                    if token_id == self.tokenizer.unk_token_id:
                        start, end = offset_mapping
                        token = sample[start:end]
                        all_unknown_tokens.append(token)
                        unknown_token_indexes.setdefault(token, []).append(idx)
                    if token_id not in non_text_token_ids:
                        total_tokens += 1
        else:
            # Tokenization for each word
            # --------------------------
            # Choose tokenizer based on availability of nltk
            if nltk.download('punkt', quiet=True):
                tokenize = nltk.word_tokenize
            else:
                warnings.warn('nltk punkt is not available, using str.split instead to identify individual words. '
                              'Please check your internet connection.')
                tokenize = str.split

            # Tokenize samples and count unknown words
            words_array = [tokenize(sample) for sample in samples]
            for idx, words in zip(indices, words_array):
                total_tokens += len(words)
                for word in words:
                    tokens = self.tokenizer.tokenize(word)
                    if any(self.tokenizer.convert_tokens_to_ids(token) == self.tokenizer.unk_token_id for token in
                           tokens):
                        all_unknown_tokens.append(word)
                        unknown_token_indexes.setdefault(word, []).append(idx)

        return Counter(all_unknown_tokens), total_tokens, unknown_token_indexes

    def create_pie_chart(self, all_unknown_words_counter, total_words):
        """Create pie chart with most common unknown words."""
        most_common_unknown_words = [x[0] for x in all_unknown_words_counter.most_common(self.n_most_common) if
                                     ((x[1] > 1) or (not self.group_singleton_words))]
        other_words = [x for x in all_unknown_words_counter if x not in most_common_unknown_words]

        # Calculate percentages for each category
        other_words_count = sum(all_unknown_words_counter[word] for word in other_words)
        other_words_percentage = (other_words_count * 1. / total_words) * 100.
        labels = most_common_unknown_words
        percentages = [all_unknown_words_counter[word] * 1. / total_words * 100. for word in most_common_unknown_words]

        # Add "Other Unknown Words" and "Known Words" categories
        if other_words_percentage > 0:
            labels.append(OTHER_CAT_NAME)
            percentages.append(other_words_percentage)

        # Truncate labels for display
        labels = [truncate_string(label, self.max_text_length_for_display) for label in labels]
        # round percentages to 2 decimal places after the percent
        percentages = [round_sig(percent, 2) for percent in percentages]

        # Create pie chart with hover text and custom hover template
        fig = go.Figure(data=[go.Pie(
            labels=labels, values=percentages, texttemplate='%{label}<br>%{value}%',
            hovertext=[format_list(other_words, max_string_length=self.max_text_length_for_display)
                       if label == OTHER_CAT_NAME else label for label in labels],
            hovertemplate=['<b>Unknown Word</b>: %{hovertext}<br><b>Percent of All Words</b>: %{value}%<extra></extra>'
                           if label != OTHER_CAT_NAME else
                           '<b>Other Unknown Words</b>: %{hovertext}<br>'
                           '<b>Percent of All Words</b>: %{value}%<extra></extra>'
                           for label in labels],
            pull=[0.1 if label == OTHER_CAT_NAME else 0 for label in labels]
        )])

        # Customize chart appearance
        fig.update_layout(title=f'Words containing Unknown Tokens - {self.tokenizer.name_or_path} Tokenizer<br>'
                                f'({format_percent(sum(percentages) / 100.)} of all words)',
                          title_x=0.5,
                          title_y=0.95,
                          legend_title='Words with Unknown Tokens',
                          margin=dict(l=0, r=0, t=100, b=0))

        return fig

    def add_condition_ratio_of_unknown_words_less_or_equal(self, ratio: float = 0):
        """Add condition that checks if the ratio of unknown words is less than a given ratio.

        Parameters
        ----------
        ratio : float
            Maximal allowed ratio of unknown words.
        """
        def condition(result):
            passed = result['unknown_word_ratio'] <= ratio
            condition_result = ConditionCategory.FAIL if not passed else ConditionCategory.PASS
            details = f'Ratio was {format_percent(result["unknown_word_ratio"])}'
            return ConditionResult(condition_result, details)

        return self.add_condition(f'Ratio of unknown words is less than {format_percent(ratio)}',
                                  condition)


class PrintFilter:
    """Filter to avoid printing of tokenization warnings."""

    def __init__(self, original_stdout):
        self.original_stdout = original_stdout

    def write(self, msg):
        if 'huggingface/tokenizers' not in msg:
            self.original_stdout.write(msg)

    def flush(self):
        self.original_stdout.flush()
