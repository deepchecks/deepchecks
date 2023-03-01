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

__all__ = ['calculate_default_properties',
           'text_length',
           'average_word_length',
           'percentage_special_characters']


def text_length(raw_text: Sequence[str]) -> List[int]:
    """Return list of integers of text lengths."""
    return [len(text) for text in raw_text]


def word_length(raw_text: Sequence[str]) -> List[int]: # Not yet used as returns list per sample and not number
    """Return list of integers of word lengths."""
    return [len(word) for text in raw_text for word in text.split()]


def average_word_length(raw_text: Sequence[str]) -> List[float]:
    """Return list of floats of average word length."""
    return [np.mean([len(word) for word in text.split()]) for text in raw_text]


def percentage_special_characters(raw_text: Sequence[str]) -> List[float]:
    """Return list of floats of percentage of special characters."""
    return [len([c for c in text if c in string.punctuation]) / len(text) for text in raw_text]


def get_language_detection() -> callable:
    """Return a function that returns the language of a text.

    As identifying the language of a text requires the langdetect package, which is not necessary for the rest of the
    NLP module, we import it only when needed.
    Therefore, the language property is not given directly, but rather through this function, so that the user will only
    get an ImportError if they try to use the language property.
    """
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


def get_sentiment_detection() -> callable:
    """Return a function that returns the language of a text.

    As identifying the sentiment of a text requires the textblob package, which is not necessary for the rest of the
    NLP module, we import it only when needed.
    Therefore, the sentiment property is not given directly, but rather through this function, so that the user will only
    get an ImportError if they try to use the sentiment property.
    """
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

def get_subjectivity_detection() -> callable:
    """Return a function that returns the language of a text.

    As identifying the subjectivity of a text requires the textblob package, which is not necessary for the rest of the
    NLP module, we import it only when needed.
    Therefore, the subjectivity property is not given directly, but rather through this function, so that the user will only
    get an ImportError if they try to use the subjectivity property.
    """

    try:
        import textblob
    except ImportError as e:
        raise ImportError(
            'property subjectivity requires the textblob python package. '
            'To get it, run "pip install textblob".') from e

    def subjectivity(raw_text: Sequence[str]) -> List[float]:
        """Return list of floats of subjectivity."""
        return [textblob.TextBlob(text).sentiment.subjectivity for text in raw_text]

    return subjectivity


def get_topic_detection(batch_size: int = 128) -> callable:
    """Return a function that returns the language of a text.

    As identifying the topic of a text requires the transformers package, which is not necessary for the rest of the
    NLP module, we import it only when needed.
    Therefore, the topic property is not given directly, but rather through this function, so that the user will only
    get an ImportError if they try to use the topic property.

    Params:
        batch_size : int, default 100
            The number of samples to process in each batch.


    """
    try:
        from transformers import AutoModelForSequenceClassification
        from transformers import AutoTokenizer
    except ImportError as e:
        raise ImportError(
            'property topic requires the transformers python package. '
            'To get it, run "pip install transformers".') from e

    MODEL = f"cardiffnlp/tweet-topic-21-multi"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    # PT
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    model = model.eval()
    class_mapping = model.config.id2label

    def topic(raw_text: Sequence[str]) -> List[float]:
        """Return list of strings of topic."""
        all_topics = []

        for i in range(0, len(raw_text), batch_size):
            tokens = tokenizer(raw_text[i:i+batch_size], return_tensors='pt', padding=True, truncation=True, max_length=128)
            output = model(**tokens)
            scores = output[0].detach().numpy()

            all_topics += [class_mapping[i] for i in np.argmax(scores, axis=1)]

        return all_topics

    return topic


def _get_default_properties(include_properties: List[str] = None, ignore_properties: List[str] = None):
    """Return the default properties.

    Default properties are defined here and not outside the function so not to import all the packages
    if they are not needed.
    """

    ret = [
        {'name': 'text_length', 'method': text_length, 'output_type': 'numeric'},
        {'name': 'average_word_length', 'method': average_word_length, 'output_type': 'numeric'},
        {'name': 'percentage_special_characters', 'method': percentage_special_characters, 'output_type': 'numeric'},
    ]

    # Filter by properties or ignore_properties:
    if include_properties is not None and ignore_properties is not None:
        raise ValueError('Cannot use properties and ignore_properties parameters together.')
    elif include_properties is not None:
        ret = [prop for prop in ret if prop['name'] in include_properties]
    elif ignore_properties is not None:
        ret = [prop for prop in ret if prop['name'] not in ignore_properties]

    # Add properties that require additional packages:
    include_properties = include_properties or []
    ignore_properties = ignore_properties or []
    if 'language' in include_properties or 'language' not in ignore_properties:
        ret.append({'name': 'language', 'method': get_language_detection(), 'output_type': 'categorical'})
    if 'sentiment' in include_properties or 'sentiment' not in ignore_properties:
        ret.append({'name': 'sentiment', 'method': get_sentiment_detection(), 'output_type': 'numeric'})
    if 'subjectivity' in include_properties or 'subjectivity' not in ignore_properties:
        ret.append({'name': 'subjectivity', 'method': get_subjectivity_detection(), 'output_type': 'numeric'})
    if 'topic' in include_properties or 'topic' not in ignore_properties:
        ret.append({'name': 'topic', 'method': get_topic_detection(), 'output_type': 'categorical'})

    return ret


def calculate_default_properties(raw_text: Sequence[str], include_properties: List[str] = None,
                                 ignore_properties: List[str] = None) -> Dict[str, List[float]]:
    """Return list of dictionaries of text properties.

    Params:
        raw_text : Sequence[str]
            The text to calculate the properties for.
        include_properties : List[str], default None
            The properties to calculate. If None, all default properties will be calculated. Cannot be used together with
            ignore_properties parameter.
        ignore_properties : List[str], default None
            The properties to ignore. If None, no properties will be ignored. Cannot be used together with
            properties parameter.

    Returns:
        Dict[str, List[float]]
            A dictionary with the property name as key and a list of the property values for each text as value.
    """
    default_text_properties = _get_default_properties(include_properties=include_properties,
                                                      ignore_properties=ignore_properties)

    return {prop['name']: prop['method'](raw_text) for prop in default_text_properties}
