"""String functions."""
from collections import defaultdict
from typing import Dict, Set

__all__ = ['string_baseform', 'get_base_form_to_variants_dict', 'underscore_to_capitalize']

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


def get_base_form_to_variants_dict(uniques):
    """Create dict of base-form of the uniques to their values.

    function gets a set of strings, and returns a dictionary of shape Dict[str]=Set,
    the key being the "base_form" (a clean version of the string),
    and the value being a set of all existing original values.
    This is done using the StringCategory class.
    """
    base_form_to_variants: Dict[str, Set] = defaultdict(set)
    for item in uniques:
        base_form_to_variants[string_baseform(item)].add(item)
    return base_form_to_variants
