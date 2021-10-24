"""String functions."""

__all__ = ['string_baseform', 'split_and_keep', 'split_and_keep_by_many']

from copy import copy
from typing import List

SPECIAL_CHARS: str = ' !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\n'


def string_baseform(string: str):
    """Remove special characters from given string.

    Args:
        string (str): string to remove special characters from

    Returns:
        (str): string without special characters
    """
    if not isinstance(string, str):
        return string
    return string.translate(str.maketrans('', '', SPECIAL_CHARS)).lower()


def underscore_to_capitalize(string: str):
    """Replace underscore with space and capitalize first letters in each word.

    Args:
        string (str): string to change
    """
    return ' '.join([s.capitalize() for s in string.split('_')])


def split_and_keep(s: str, separator: str) -> List[str]:
    """
    Split string by a another substring into a list. Like str.split(), but keeps the separator occurrences in the list.

    Args:
        s (str): the string to split
        separator (str): the substring to split by

    Returns:
        List[str]: list of substrings, including the separator occurrences in string

    """
    split_s = []
    while len(s) != 0:
        if s.find(separator) == 0:
            split_s.append(separator)
            s = s[len(separator):]
        else:
            pre, _ = s.split(separator, 1)
            split_s.append(pre)
            s = s[len(pre):]
    return split_s


def split_and_keep_by_many(s: str, separators: List[str], keep: bool = True) -> List[str]:
    """
    Split string by a a list of substrings, each used once as a separator.

    Args:
        s (str): the string to split
        separators (List[str]): list of substrings to split by
        keep (bool): whether to keep the separators in list as well. Default is True.

    Returns:
        List[str]: list of substrings
    """
    split_s = []
    separators = copy(separators)
    while len(s) != 0:
        if len(separators) > 0:
            sep = separators[0]
            if s.find(sep) == 0:
                if keep is True:
                    split_s.append(sep)
                s = s[len(sep):]
                separators.pop(0)
            else:
                pre, _ = s.split(sep, 1)
                split_s.append(pre)
                s = s[len(pre):]
        else:
            split_s.append(s)
            break
    return split_s
