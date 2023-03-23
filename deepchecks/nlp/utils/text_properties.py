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
"""Module containing the text properties for the NLP module."""
import string
import warnings
from typing import Dict, List, Sequence, Tuple

import numpy as np

__all__ = ['calculate_default_properties']


def property_import_error(property_name: str, package_name: str) -> ImportError:
    """Raise an ImportError for a property that requires a package."""
    return ImportError(
        f'property {property_name} requires the {package_name} python package. '
        f'To get it, run "pip install {package_name}".')


def text_length(raw_text: Sequence[str]) -> List[int]:
    """Return list of integers of text lengths."""
    return [len(text) for text in raw_text]


def word_length(raw_text: Sequence[str]) -> List[int]:  # Not yet used as returns list per sample and not number
    """Return list of integers of word lengths."""
    return [len(word) for text in raw_text for word in text.split()]


def average_word_length(raw_text: Sequence[str]) -> List[float]:
    """Return list of floats of average word length."""
    return [np.mean([len(word) for word in text.split()]) for text in raw_text]


def percentage_special_characters(raw_text: Sequence[str]) -> List[float]:
    """Return list of floats of percentage of special characters."""
    return [len([c for c in text if c in string.punctuation]) / len(text) for text in raw_text]


def max_word_length(raw_text: Sequence[str]) -> List[int]:
    """Return list of integers of max word length."""
    return [max([len(word) for word in text.split()]) for text in raw_text]


def language(raw_text: Sequence[str]) -> List[str]:
    """Return list of strings of language."""
    try:
        import langdetect  # pylint: disable=import-outside-toplevel
        from langdetect import DetectorFactory  # pylint: disable=import-outside-toplevel
        DetectorFactory.seed = 42
    except ImportError as e:
        raise property_import_error('language', 'langdetect') from e

    return [langdetect.detect(text) for text in raw_text]


def sentiment(raw_text: Sequence[str]) -> List[str]:
    """Return list of floats of sentiment."""
    try:
        import textblob  # pylint: disable=import-outside-toplevel
    except ImportError as e:
        raise property_import_error('sentiment', 'textblob') from e

    return [textblob.TextBlob(text).sentiment.polarity for text in raw_text]


def subjectivity(raw_text: Sequence[str]) -> List[str]:
    """Return list of floats of subjectivity."""
    try:
        import textblob  # pylint: disable=import-outside-toplevel
    except ImportError as e:
        raise property_import_error('subjectivity', 'textblob') from e

    return [textblob.TextBlob(text).sentiment.subjectivity for text in raw_text]


DEFAULT_PROPERTIES = [
    {'name': 'text_length', 'method': text_length, 'output_type': 'numeric'},
    {'name': 'average_word_length', 'method': average_word_length, 'output_type': 'numeric'},
    {'name': 'percentage_special_characters', 'method': percentage_special_characters, 'output_type': 'numeric'},
    {'name': 'Max Word Lengh', 'method': max_word_length, 'output_type': 'numeric'},
    {'name': 'language', 'method': language, 'output_type': 'categorical'},
    {'name': 'sentiment', 'method': sentiment, 'output_type': 'numeric'},
    {'name': 'subjectivity', 'method': subjectivity, 'output_type': 'numeric'},
]


def _get_default_properties(include_properties: List[str] = None, ignore_properties: List[str] = None):
    """Return the default properties.

    Default properties are defined here and not outside the function so not to import all the packages
    if they are not needed.
    """
    ret_properties = DEFAULT_PROPERTIES

    # Filter by properties or ignore_properties:
    if include_properties is not None and ignore_properties is not None:
        raise ValueError('Cannot use properties and ignore_properties parameters together.')
    elif include_properties is not None:
        ret_properties = [prop for prop in ret_properties if prop['name'] in include_properties]
    elif ignore_properties is not None:
        ret_properties = [prop for prop in ret_properties if prop['name'] not in ignore_properties]

    return ret_properties


def calculate_default_properties(raw_text: Sequence[str], include_properties: List[str] = None,
                                 ignore_properties: List[str] = None) -> Tuple[Dict[str, List[float]], Dict[str, str]]:
    """Return list of dictionaries of text properties.

    Params:
        raw_text : Sequence[str]
            The text to calculate the properties for.
        include_properties : List[str], default None
            The properties to calculate. If None, all default properties will be calculated. Cannot be used together
            with ignore_properties parameter.
        ignore_properties : List[str], default None
            The properties to ignore. If None, no properties will be ignored. Cannot be used together with
            properties parameter.

    Returns:
        Dict[str, List[float]]
            A dictionary with the property name as key and a list of the property values for each text as value.
    """
    default_text_properties = _get_default_properties(include_properties=include_properties,
                                                      ignore_properties=ignore_properties)

    calculated_properties = {}
    for prop in default_text_properties:
        try:
            res = prop['method'](raw_text)
            calculated_properties[prop['name']] = res
        except ImportError as e:
            warnings.warn(f'Failed to calculate property {prop["name"]}. Error: {e}')
    if not calculated_properties:
        raise RuntimeError('Failed to calculate any of the properties.')

    properties_types = {prop['name']: prop['output_type'] for prop in default_text_properties}  # TODO: Add tests

    return calculated_properties, properties_types
