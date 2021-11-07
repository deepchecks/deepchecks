"""String functions."""
import re
from collections import defaultdict
from decimal import Decimal
from typing import Dict, Set, List, Union, Tuple
from copy import copy
import pandas as pd


__all__ = ['string_baseform', 'get_base_form_to_variants_dict', 'split_capitalized', 'split_and_keep',
           'split_by_order', 'is_string_column', 'format_percent', 'format_number']

from pandas.core.dtypes.common import is_numeric_dtype


def string_baseform(string: str):
    """Remove special characters from given string, leaving only a-z, A-Z, 0-9 characters.

    Args:
        string (str): string to remove special characters from

    Returns:
        (str): string without special characters
    """
    if not isinstance(string, str):
        return string
    return re.sub('[^A-Za-z0-9]+', '', string).lower()


def is_string_column(column: pd.Series):
    """Determine whether a pandas series is string type."""
    if is_numeric_dtype(column):
        return False
    try:
        pd.to_numeric(column)
        return False
    except ValueError:
        return True


def split_capitalized(string: str):
    """Replace underscore with space and capitalize first letters in each word.

    Args:
        string (str): string to change
    """
    return ' '.join(re.findall('[A-Z][^A-Z]*', string))


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


def str_min_find(s: str, substr_list: List[str]) -> Tuple[int, str]:
    """
    Find the minimal first occurence of a substring in a string, and return both the index and substring.

    Args:
        s (str): The string in which we look for substrings
        substr_list: list of substrings to find

    Returns:
        min_find (int): index of minimal first occurence of substring
        min_substr (str): the substring that occures in said index

    """
    min_find = -1
    min_substr = ''
    for substr in substr_list:
        first_find = s.find(substr)
        if first_find != -1 and (first_find < min_find or min_find == -1):
            min_find = first_find
            min_substr = substr
    return min_find, min_substr


def split_and_keep(s: str, separators: Union[str, List[str]]) -> List[str]:
    """
    Split string by a another substring into a list. Like str.split(), but keeps the separator occurrences in the list.

    Args:
        s (str): the string to split
        separators (str): the substring to split by

    Returns:
        List[str]: list of substrings, including the separator occurrences in string

    """
    if isinstance(separators, str):
        separators = [separators]

    split_s = []
    while len(s) != 0:
        i, substr = str_min_find(s=s, substr_list=separators)
        if i == 0:
            split_s.append(substr)
            s = s[len(substr):]
        elif i == -1:
            split_s.append(s)
            break
        else:
            pre, _ = s.split(substr, 1)
            split_s.append(pre)
            s = s[len(pre):]
    return split_s


def split_by_order(s: str, separators: List[str], keep: bool = True) -> List[str]:
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


def format_percent(ratio: float, floating_point: int = 2) -> str:
    """Format percent for elegant display.

    Args:
        ratio (float): Number [0-1] to be displayed as percent
        floating_point (int): Number of floating points to display

    Returns:
        String of ratio as percent
    """
    if (ratio > 1) or (ratio < 0):
        Exception(ValueError('ratio must be between 0 and 1'))
    if ratio == 0:
        return '0%'
    if ratio == 1:
        return '100%'
    if ratio < 10**(-(2+floating_point)):
        return f'{Decimal(ratio * 100):.{floating_point}E}%'
    elif ratio > (1-10**(-(2+floating_point))):
        if floating_point > 0:
            return f'99.{"".join(["9"]*floating_point)}%'
        else:
            return '99%'
    else:
        return f'{ratio:.{floating_point}%}'


def format_number(x, floating_point: int = 2) -> str:
    """Format number for elegant display.

    Args:
        x (): Number to be displayed
        floating_point (int): Number of floating points to display

    Returns:
        String of beautified number
    """
    def add_commas(x):
        return f'{x:,}'  # yes this actually formats the number 1000 to "1,000"

    # 0 is lost in the next if case, so we have it here as a special use-case
    if x == 0:
        return '0'

    # If x is a very small number, that would be rounded to 0, we would prefer to return it as the format 1.0E-3.
    if abs(x) < 10 ** (-floating_point):
        return f'{Decimal(x):.{floating_point}E}'

    # If x is an integer, or if x when rounded is an integer (e.g. 1.999999), then return as integer:
    if round(x) == round(x, floating_point):
        return add_commas(round(x))

    # If not, return as a float, but don't print unnecessary zeros at end:
    else:
        ret_x = round(x, floating_point)
        return add_commas(ret_x).rstrip('0')
