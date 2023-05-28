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
"""Module of text utils for NLP package."""
import re
import string
import typing as t
import unicodedata
import warnings

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

__all__ = [
    'break_to_lines_and_trim',
    'normalize_text',
    'hash_text',
    'normalize_samples',
    'hash_samples'
]


def break_to_lines_and_trim(s, max_lines: int = 10, min_line_length: int = 50, max_line_length: int = 60):
    """Break a string to lines and trim it to a maximum number of lines.

    Parameters
    ----------
    s : str
        The string to break.
    max_lines : int, default 10
        The maximum number of lines to return.
    min_line_length : int, default 50
        The minimum length of a line.
    max_line_length : int, default 60
        The maximum length of a line.
    """
    separating_delimiters = [' ', '\t', '\n', '\r']
    lines = []
    for i in range(max_lines):  # pylint: disable=unused-variable
        if len(s) < max_line_length:  # if remaining string is short enough, add it and break
            lines.append(s.strip())
            break
        else:  # find the first delimiter from the end of the line
            max_line_length = min(max_line_length, len(s)-1)
            for j in range(max_line_length, min_line_length-1, -1):
                if s[j] in separating_delimiters:
                    lines.append(s[:j])
                    s = s[j:].strip()
                    break
            else:  # if no delimiter was found, break in the middle of the line
                # Check if breaking in the middle of an HTML tag
                tag_start = re.search(r'<[^>]*$', s[:max_line_length])
                if tag_start:
                    max_line_length = tag_start.start()
                lines.append(s[:max_line_length].strip() + '-')
                s = s[max_line_length:].strip()
    else:  # if the loop ended without breaking, and there is still text left, add an ellipsis
        if len(s) > 0:
            lines[-1] = lines[-1] + '...'
    return '<br>'.join(lines)


def remove_punctuation(text: str) -> str:
    """Remove punctuation characters from a string."""
    return text.translate(str.maketrans('', '', string.punctuation))


def normalize_unicode(text: str) -> str:
    """Normalize unicode characters."""
    return unicodedata.normalize('NFKC', text)


def remove_stopwords(text: str) -> str:
    """Remove stop words from a string."""
    if nltk.download('stopwords', quiet=True):
        stop_words = set(stopwords.words('english'))
    else:
        warnings.warn('nltk stopwords not found, stopwords won\'t be ignored when considering text duplicates.'
                      ' Please check your internet connection.')
        return text
    if nltk.download('punkt', quiet=True):
        tokenize = word_tokenize
    else:
        tokenize = str.split
    words = tokenize(text)
    return ' '.join([word for word in words if word.lower() not in stop_words])


def normalize_text(
    text_sample: str,
    *,
    ignore_case: bool = True,
    remove_punct: bool = True,
    normalize_uni: bool = True,
    remove_stops: bool = True,
    ignore_whitespace: bool = False
) -> str:
    """Normalize given text sample."""
    if ignore_case:
        text_sample = text_sample.lower()
    if remove_punct:
        text_sample = remove_punctuation(text_sample)
    if normalize_uni:
        text_sample = normalize_unicode(text_sample)
    if remove_stops:
        text_sample = remove_stopwords(text_sample)
    if ignore_whitespace:
        text_sample = ''.join(text_sample.split())
    return text_sample


def cut_string(input_str: str, cut_length: int = 200) -> str:
    """Cut a string to 200 characters, but cut only at whitespaces."""
    if len(input_str) > cut_length:
        index = input_str.find(' ', cut_length)
        if index != -1:
            return input_str[:index]
    return input_str


def normalize_samples(
    text_samples: t.Sequence[str],
    *,
    ignore_case: bool = True,
    remove_punct: bool = True,
    normalize_uni: bool = True,
    remove_stops: bool = True,
    ignore_whitespace: bool = False
) -> t.List[str]:
    """Normalize given sequence of text samples."""
    return [
        normalize_text(
            it,
            ignore_case=ignore_case,
            remove_punct=remove_punct,
            normalize_uni=normalize_uni,
            remove_stops=remove_stops,
            ignore_whitespace=ignore_whitespace
        )
        for it in text_samples
    ]


def hash_text(text: str) -> int:
    """Hash a text sample."""
    assert isinstance(text, str)
    return hash(text)


def hash_samples(text: t.Sequence[str]) -> t.List[int]:
    """Hash a sequence of text samples."""
    assert not isinstance(text, str)
    return [hash_text(it) for it in text]
