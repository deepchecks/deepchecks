# ----------------------------------------------------------------------------
# Copyright (C) 2021-2022 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""Module containing the text properties for the NLP module."""
import string
from typing import List, Sequence, Dict

import numpy as np

__all__ = ['default_text_properties',
           'text_length',
           'average_word_length',
           'percentage_special_characters']

def text_length(raw_text: Sequence[str]) -> List[int]:
    """Return list of integers of text lengths."""
    return [len(text) for text in raw_text]

def word_length(raw_text: Sequence[str]) -> List[int]:
    """Return list of integers of word lengths."""
    return [len(word) for text in raw_text for word in text.split()]

def average_word_length(raw_text: Sequence[str]) -> List[float]:
    """Return list of floats of average word length."""
    return [np.mean([len(word) for word in text.split()]) for text in raw_text]

def percentage_special_characters(raw_text: Sequence[str]) -> List[float]:
    """Return list of floats of percentage of special characters."""
    return [len([c for c in text if c in string.punctuation]) / len(text) for text in raw_text]


def get_language_detection() -> callable:
    try:
        import langdetect
    except ImportError as e:
        raise ImportError(
            'property language requires the langdetect python package. '
            'To get it, run "pip install langdetect".') from e

    def language(raw_text: Sequence[str]) -> List[str]:
        """Return list of strings of language."""
        return [langdetect.detect(text) for text in raw_text]

    return language

# def get_language_detection() -> callable:
#     try:
#         import bertopic
#     except ImportError as e:
#         raise ImportError(
#             'property language requires the langdetect python package. '
#             'To get it, run "pip install bertopic".') from e
#
#     def language(raw_text: Sequence[str]) -> List[str]:
#         """Return list of strings of language."""
#         return [langdetect.detect(text) for text in raw_text]
#
#     return language

def get_sentiment_detection() -> callable:
    try:
        import textblob
    except ImportError as e:
        raise ImportError(
            'property sentiment requires the textblob python package. '
            'To get it, run "pip install textblob".') from e

    def sentiment(raw_text: Sequence[str]) -> List[float]:
        """Return list of floats of sentiment."""
        return [textblob.TextBlob(text).sentiment.polarity for text in raw_text]

    return sentiment

def get_topic_detection() -> callable:
    try:
        from transformers import AutoModelForSequenceClassification
        from transformers import AutoTokenizer
    except ImportError as e:
        raise ImportError(
            'property sentiment requires the transformers python package. '
            'To get it, run "pip install transformers".') from e

    MODEL = f"cardiffnlp/tweet-topic-21-multi"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    # PT
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    model = model.eval()
    class_mapping = model.config.id2label

    def topic(raw_text: Sequence[str]) -> List[float]:
        """Return list of floats of sentiment."""
        all_topics = []

        for i in range(0, len(raw_text), 100):
            tokens = tokenizer(raw_text[i:i+100], return_tensors='pt', padding=True)
            output = model(**tokens)
            scores = output[0].detach().numpy()

            all_topics += [class_mapping[i] for i in np.argmax(scores, axis=1)]

        return all_topics

        # return [class_mapping[i] for i in np.argmax(scores, axis=1)]

    return topic


default_text_properties = [
    {'name': 'text_length', 'method': text_length, 'output_type': 'numeric'},
    # {'name': 'word_length', 'method': word_length, 'output_type': 'numeric'},
    # {'name': 'average_word_length', 'method': average_word_length, 'output_type': 'numeric'},
    # {'name': 'percentage_special_characters', 'method': percentage_special_characters, 'output_type': 'numeric'},
    # {'name': 'language', 'method': get_language_detection(), 'output_type': 'categorical'},
    {'name': 'sentiment', 'method': get_sentiment_detection(), 'output_type': 'numeric'},
    {'name': 'topic', 'method': get_topic_detection(), 'output_type': 'categorical'},
]



def calculate_default_properties(raw_text: Sequence[str]) -> Dict[str, List[float]]:
    """Return list of dictionaries of text properties."""
    return {prop['name']: prop['method'](raw_text) for prop in default_text_properties}

def get_default_property_types() -> Dict[str, str]:
    """Return dictionary of default property types."""
    return {prop['name']: prop['output_type'] for prop in default_text_properties}