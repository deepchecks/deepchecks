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
import string
import typing as t
from collections import Counter

import nltk
import pandas as pd
from transformers import BertTokenizer
from typing_extensions import Self, TypedDict
import plotly.graph_objects as go

from deepchecks.core import CheckResult, ConditionCategory, ConditionResult
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.nlp import Context, SingleDatasetCheck
from deepchecks.nlp._shared_docs import docstrings
from deepchecks.nlp.text_data import TextData
from deepchecks.utils.strings import SPECIAL_CHARACTERS, format_list, format_percent
from deepchecks.utils.strings import get_ellipsis as truncate_string

__all__ = ['UnknownTokens']


@docstrings
class UnknownTokens(SingleDatasetCheck):
    """Find samples that contain tokens unsupported by your tokenizer.

    Parameters
    ----------
    tokenizer: t.Any , default: None
        Transformers tokenizer to use for tokenization. If None, BertTokenizer.from_pretrained('bert-base-uncased')
        will be used.
    n_most_common : int , default: 5
        Number of most common words with unknown tokens to show in results
    n_samples: int, default: 1_000_000
        number of samples to use for this check.
    random_state : int, default: 42
        random seed for all check internals.
    {max_text_length_for_display_param:1*indent}
    """

    def __init__(
        self,
        tokenizer: t.Any = None,
        n_most_common: int = 5,
        n_samples: int = 1_000_000,
        random_state: int = 42,
        max_text_length_for_display: int = 30,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        if self.tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        if not hasattr(self.tokenizer, 'tokenize'):
            raise DeepchecksValueError('tokenizer must have a "tokenize" method')
        self.n_most_common = n_most_common
        self.n_samples = n_samples
        self.random_state = random_state
        self.max_text_length_for_display = max_text_length_for_display

    def run_logic(self, context: Context, dataset_kind) -> CheckResult:
        """Run check."""
        dataset = context.get_data_by_kind(dataset_kind).sample(self.n_samples, random_state=self.random_state)
        dataset = t.cast(TextData, dataset)
        samples = dataset.text
        if len(samples) == 0:
            raise DeepchecksValueError('Dataset cannot be empty')

        all_unknown_words_counter, total_words = self.find_unknown_words(samples)
        if len(all_unknown_words_counter) == 0:
            display = None
        else:
            fig = self.create_pie_chart(all_unknown_words_counter, total_words)
            display = [fig]

        return CheckResult(None, display=display)

    def find_unknown_words(self, samples):
        if nltk.download('punkt'):
            tokenize = nltk.word_tokenize
        else:
            tokenize = str.split

        words_array = [tokenize(sample) for sample in samples]

        all_unknown_words = []
        total_words = 0

        for words in words_array:
            total_words += len(words)
            for word in words:
                tokens = self.tokenizer.tokenize(word)
                if any(self.tokenizer.convert_tokens_to_ids(token) == self.tokenizer.unk_token_id for token in tokens):
                    all_unknown_words.append(word)

        return Counter(all_unknown_words), total_words

    def create_pie_chart(self, all_unknown_words_counter, total_words):
        most_common_unknown_words = [x[0] for x in all_unknown_words_counter.most_common(self.n_most_common) if
                                     x[1] > 1]
        other_words = [x for x in all_unknown_words_counter if x not in most_common_unknown_words]

        other_words_count = sum(all_unknown_words_counter[word] for word in other_words)
        other_words_percentage = (other_words_count / total_words) * 100

        labels = most_common_unknown_words
        percentages = [all_unknown_words_counter[word] / total_words * 100 for word in most_common_unknown_words]

        if other_words_percentage > 0:
            labels.append('Other Unknown Words')
            percentages.append(other_words_percentage)

        known_words_percentage = 100 - sum(percentages)
        labels.append('Known Words')
        percentages.append(known_words_percentage)

        labels = [truncate_string(label, self.max_text_length_for_display) for label in labels]

        fig = go.Figure(data=[go.Pie(
            labels=labels, values=percentages, textinfo='label+percent',
            hovertext=[' '.join(other_words[:self.max_text_length_for_display]) if label == 'Other Unknown Words'
                       else label for label in labels],
            hovertemplate='%{hovertext}<br>%{percent}<extra></extra>',
            pull=[0.1 if label != 'Known Words' else 0 for label in labels]
        )])

        fig.update_layout(title='Words Containing Unknown Tokens', legend_title='Words')

        return fig
