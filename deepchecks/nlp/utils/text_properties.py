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
from typing import List

import numpy as np

__all__ = ['default_text_properties',
           'text_length',
           'average_word_length',
           'percentage_special_characters']

def text_length(raw_text: List[str]) -> List[int]:
    """Return list of integers of text lengths."""
    return [len(text) for text in raw_text]

def word_length(raw_text: List[str]) -> List[int]:
    """Return list of integers of word lengths."""
    return [len(word) for text in raw_text for word in text.split()]

def average_word_length(raw_text: List[str]) -> List[float]:
    """Return list of floats of average word length."""
    return [np.mean([len(word) for word in text.split()]) for text in raw_text]

def percentage_special_characters(raw_text: List[str]) -> List[float]:
    """Return list of floats of percentage of special characters."""
    return [len([c for c in text if c in string.punctuation]) / len(text) for text in raw_text]


default_text_properties = [
    {'name': 'text_length', 'method': text_length, 'output_type': 'numeric'},
    # {'name': 'word_length', 'method': word_length, 'output_type': 'numeric'},
    {'name': 'average_word_length', 'method': average_word_length, 'output_type': 'numeric'},
    {'name': 'percentage_special_characters', 'method': percentage_special_characters, 'output_type': 'numeric'},
]

def calculate_properties(raw_text: List[str], properties: List[dict] = default_text_properties) -> List[dict]:
    """Return list of dictionaries of text properties."""
    return [{prop['name']: prop['method'](raw_text)} for prop in properties]