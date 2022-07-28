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
"""String functions."""
import io
import itertools
import json
import os
import random
import re
import sys
import typing as t
from collections import defaultdict
from copy import copy
from datetime import datetime
from decimal import Decimal
from string import ascii_uppercase, digits

import numpy as np
import pandas as pd
from ipywidgets import Widget
from ipywidgets.embed import dependency_state, embed_data, escape_script, snippet_template, widget_view_template
from packaging.version import Version
from pandas.core.dtypes.common import is_numeric_dtype

import deepchecks
from deepchecks import core
from deepchecks.core.resources import jupyterlab_plotly_script, requirejs_script, suite_template, widgets_script
from deepchecks.utils.typing import Hashable

__all__ = [
    'string_baseform',
    'get_base_form_to_variants_dict',
    'split_camel_case',
    'split_and_keep',
    'split_by_order',
    'is_string_column',
    'format_percent',
    'format_number',
    'format_list',
    'get_random_string',
    'format_datetime',
    'get_docs_summary',
    'get_ellipsis',
    'to_snake_case',
    'create_new_file_name',
    'widget_to_html',
    'generate_check_docs_link',
    'widget_to_html_string',
    'format_number_if_not_nan',
    'get_docs_link'
]

# Creating a translation table for the string.translate() method to be used in string base-form method
DEL_CHARS = ''.join(c for c in map(chr, range(sys.maxunicode)) if not c.isalnum())
DEL_MAP = str.maketrans('', '', DEL_CHARS)


def get_ellipsis(long_string: str, max_length: int):
    """Return the long string with ellipsis if above max_length.

    Parameters
    ----------
    long_string : str
        the string
    max_length : int
        the string maximum length

    Returns
    -------
    str
        the string with ellipsis.
    """
    if len(long_string) <= max_length:
        return long_string
    return long_string[:max_length] + '...'


def get_docs_summary(obj, with_doc_link: bool = True):
    """Return the docs summary if available.

    Parameters
    ----------
    obj
        an object
    with_doc_link : bool , default: True
        if to add doc link
    Returns
    -------
    str
        the object summary.
    """
    if hasattr(obj.__class__, '__doc__'):
        docs = obj.__class__.__doc__ or ''
        # Take first non-whitespace line.
        summary = next((s for s in docs.split('\n') if not re.match('^\\s*$', s)), '')

        if with_doc_link:
            link = generate_check_docs_link(obj)
            summary += f' <a href="{link}" target="_blank">Read More...</a>'

        return summary
    return ''


def widget_to_html(
    widget: Widget,
    html_out: t.Union[str, io.TextIOWrapper],
    title: str = '',
    requirejs: bool = True,
    connected: bool = True,
    full_html: bool = True
):
    """Save widget as html file.

    Parameters
    ----------
    widget: Widget
        The widget to save as html.
    html_out: filename or file-like object
        The file to write the HTML output to.
    title: str , default: None
        The title of the html file.
    requirejs: bool , default: True
        If to save with all javascript dependencies.
    connected : bool, default True
        whether to use CDN or not
    full_html: bool, default True
        whether to return full html page or not
    """
    state = dependency_state(widget)
    data = embed_data(views=[widget], drop_defaults=True, state=state)

    snippet = snippet_template.format(
        load='',  # will be added below
        json_data=escape_script(json.dumps(data['manager_state'])),
        widget_views='\n'.join(
            widget_view_template.format(view_spec=escape_script(json.dumps(view_spec)))
            for view_spec in data['view_specs']
        )
    )

    template = suite_template(full_html=full_html)
    html = template.replace('$Title', title).replace('$WidgetSnippet', snippet)

    # if connected is True widgets js library will load jupyterlab-plotly by itself
    jupyterlab_plotly_lib = jupyterlab_plotly_script(False) if connected is False else ''

    requirejs_lib = requirejs_script(connected) if requirejs else ''
    widgetsjs_lib = widgets_script(connected, amd_module=requirejs)
    tags = f'{requirejs_lib}{jupyterlab_plotly_lib}{widgetsjs_lib}'
    html = html.replace('$WidgetJavascript', tags)

    if isinstance(html_out, str):
        with open(html_out, 'w', encoding='utf-8') as f:
            f.write(html)
    elif isinstance(html_out, (io.TextIOBase, io.TextIOWrapper)):
        html_out.write(html)
    else:
        name = type(html_out).__name__
        raise TypeError(f'Unsupported type of "html_out" parameter - {name}')


def widget_to_html_string(
    widget: Widget,
    title: str = '',
    requirejs: bool = True,
    connected: bool = True,
    full_html: bool = True,
) -> str:
    """Transform widget into html string.

    Parameters
    ----------
    widget: Widget
        The widget to save as html.
    title: str
        The title of the html file.
    requirejs: bool , default: True
        If to save with all javascript dependencies
    connected : bool, default True
        whether to use CDN or not
    full_html: bool, default True
        whether to return full html page or not

    Returns
    -------
    str
    """
    buffer = io.StringIO()
    widget_to_html(
        widget=widget,
        html_out=buffer,
        title=title,
        requirejs=requirejs,
        connected=connected,
        full_html=full_html
    )
    buffer.seek(0)
    return buffer.getvalue()


def get_docs_link():
    """Return the link to the docs with current version.

    Returns
    -------
    str
        the link to the docs.
    """
    if deepchecks.__version__ and deepchecks.__version__ != 'dev':
        version_obj: Version = Version(deepchecks.__version__)
        # The version in the docs url is without the hotfix part
        version = f'{version_obj.major}.{version_obj.minor}'
    else:
        version = 'stable'
    return f'https://docs.deepchecks.com/{version}/'


def generate_check_docs_link(check):
    """Create from check object a link to its example page in the docs."""
    if not isinstance(check, core.BaseCheck):
        return ''

    module_path = type(check).__module__

    # NOTE:
    # it is better to import deepchecks.tabular.checks, deepchecks.vision.checks
    # to be sure that those packages actually exists and we are using right names,
    # but we do not know with what set of extra dependencies deepchecks was
    # installed, therefore we do not want to cause ImportError.
    # Refer to the setup.py for more understanding

    if not (
        module_path.startswith('deepchecks.tabular.checks')
        or module_path.startswith('deepchecks.vision.checks')
    ):
        # not builtin check, cannot generate link to the docs
        return ''

    link_postfix = '.html?utm_source=display_output&utm_medium=referral&utm_campaign=check_link'

    # compare check full name and link to the notebook to
    # understand how link is formatted:
    #
    # - deepchecks.tabular.checks.integrity.StringMismatchComparison
    # - https://docs.deepchecks.com/{version}/checks_gallery/tabular/integrity/plot_string_mismatch_comparison.html # noqa: E501 # pylint: disable=line-too-long

    # Remove 'deepchecks' from the start and 'checks' from the middle
    module_path = module_path[len('deepchecks.'):]
    module_parts = module_path.split('.')
    module_parts.remove('checks')
    # Add to the check name prefix of 'plot_'
    module_parts[-1] = f'plot_{module_parts[-1]}'
    return get_docs_link() + 'checks_gallery/' + '/'.join([*module_parts]) + link_postfix


def get_random_string(n: int = 5):
    """Return random string at the given size.

    Parameters
    ----------
    n : int , default: 5
        the size of the string to return.

    Returns
    -------
    str
        a random string
    """
    return ''.join(random.choices(ascii_uppercase + digits, k=n))


def string_baseform(string: Hashable, allow_empty_result: bool = False) -> Hashable:
    """Normalize the string input to a uniform form.

    If input is a string containing alphanumeric characters or if allow_empty_result is set to True,
    removes all non-alphanumeric characters and convert characters to lower form.

    Parameters
    ----------
    allow_empty_result : bool , default : False
        bool indicating whether to return empty result if no alphanumeric characters are present or the original input
    string : str
        string to remove special characters from

    Returns
    -------
    str
        original input if condition is not met or lower form alphanumeric characters of input.
    """
    if not isinstance(string, str):
        return string
    lower_alphanumeric_form = string.translate(DEL_MAP).lower()
    if len(lower_alphanumeric_form) > 0 or allow_empty_result:
        return lower_alphanumeric_form
    else:
        return string


def is_string_column(column: pd.Series) -> bool:
    """Determine whether a pandas series is string type."""
    if is_numeric_dtype(column):
        return False
    try:
        pd.to_numeric(column)
        return False
    except ValueError:
        return True
    # Non string objects like pd.Timestamp results in TypeError
    except TypeError:
        return False


def split_camel_case(string: str) -> str:
    """Split string where there are capital letters and enter space instead.

    Parameters
    ----------
    string : str
        string to change
    """
    return ' '.join(re.findall('[A-Z][^A-Z]*', string))


def to_snake_case(value: str) -> str:
    """Transform camel case indentifier into snake case.

    Parameters
    ----------
    value : str
        string to transform

    Returns
    -------
    str
        transformed value
    """
    return split_camel_case(value).strip().replace(' ', '_')


def get_base_form_to_variants_dict(uniques: t.Iterable[str]) -> t.Dict[str, t.Set[str]]:
    """Create dict of base-form of the uniques to their values.

    function gets a set of strings, and returns a dictionary of shape Dict[str, Set]
    the key being the "base_form" (a clean version of the string),
    and the value being a set of all existing original values.
    This is done using the StringCategory class.
    """
    base_form_to_variants = defaultdict(set)
    for item in uniques:
        base_form_to_variants[string_baseform(item)].add(item)
    return base_form_to_variants


def str_min_find(s: str, substr_list: t.Iterable[str]) -> t.Tuple[int, str]:
    """
    Find the minimal first occurence of a substring in a string, and return both the index and substring.

    Parameters
    ----------
    s : str
        The string in which we look for substrings
    substr_list : t.Iterable[str]
        list of substrings to find
    Returns
    -------
    min_find : int
        index of minimal first occurence of substring
    min_substr : str
        the substring that occures in said index

    """
    min_find = -1
    min_substr = ''
    for substr in substr_list:
        first_find = s.find(substr)
        if first_find != -1 and (first_find < min_find or min_find == -1):
            min_find = first_find
            min_substr = substr
    return min_find, min_substr


def split_and_keep(s: str, separators: t.Union[str, t.Iterable[str]]) -> t.List[str]:
    """
    Split string by a another substring into a list. Like str.split(), but keeps the separator occurrences in the list.

    Parameters
    ----------
    s : str
        the string to split
    separators : t.Union[str, t.Iterable[str]]
        the substring to split by
    Returns
    -------
    t.List[str]
        list of substrings, including the separator occurrences in string

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


def split_by_order(s: str, separators: t.Iterable[str], keep: bool = True) -> t.List[str]:
    """
    Split string by a a list of substrings, each used once as a separator.

    Parameters
    ----------
    s : str
        the string to split
    separators : t.Iterable[str]
        list of substrings to split by
    keep : bool , default: True
        whether to keep the separators in list as well. Default is True.
    Returns
    -------
    t.List[str]
        list of substrings
    """
    split_s = []
    separators = list(copy(separators))
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


def truncate_zero_percent(ratio: float, floating_point: int):
    """Display ratio as percent without trailing zeros."""
    return f'{ratio * 100:.{floating_point}f}'.rstrip('0').rstrip('.') + '%'


def format_percent(ratio: float, floating_point: int = 2, scientific_notation_threshold: int = 4,
                   add_positive_prefix: bool = False) -> str:
    """Format percent for elegant display.

    Parameters
    ----------
    ratio : float
        Ratio to be displayed as percent
    floating_point: int , default: 2
        Number of floating points to display
    scientific_notation_threshold: int, default: 4
        Max number of floating points for which to show number as float. If number of floating points is larger than
        this parameter, scientific notation (e.g. "10E-5%") will be shown.
    add_positive_prefix: bool, default: False
        add plus sign before positive percentages (minus sign is always added for negative percentages).
    Returns
    -------
    str
        String of ratio as percent
    """
    result: str
    if ratio < 0:
        ratio = -ratio
        prefix = '-'
    else:
        prefix = '+' if add_positive_prefix and ratio != 0 else ''

    if int(ratio) == ratio:
        result = f'{int(ratio) * 100}%'
    elif ratio > 1:
        result = truncate_zero_percent(ratio, floating_point)
    elif ratio < 10**(-(2+floating_point)):
        if ratio > 10**(-(2+scientific_notation_threshold)):
            result = truncate_zero_percent(ratio, scientific_notation_threshold)
        else:
            result = f'{Decimal(ratio * 100):.{floating_point}E}%'
    elif ratio > (1-10**(-(2+floating_point))):
        if floating_point > 0:
            result = f'99.{"".join(["9"]*floating_point)}%'
        else:
            result = '99%'
    else:
        result = truncate_zero_percent(ratio, floating_point)

    return prefix + result


def format_number(x, floating_point: int = 2) -> str:
    """Format number for elegant display.

    Parameters
    ----------
    x
        Number to be displayed
    floating_point : int , default: 2
        Number of floating points to display
    Returns
    -------
    str
        String of beautified number
    """
    def add_commas(x):
        return f'{x:,}'  # yes this actually formats the number 1000 to "1,000"

    if np.isnan(x):
        return 'nan'

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


def format_number_if_not_nan(x, floating_point: int = 2):
    """Format number if it is not nan for elegant display.

    Parameters
    ----------
    x
        Number to be displayed
    floating_point : int , default: 2
        Number of floating points to display
    Returns
    -------
    str
        String of beautified number if number is not nan
    """
    if np.isnan(x):
        return x
    return format_number(x, floating_point)


def format_list(l: t.List[Hashable], max_elements_to_show: int = 10, max_string_length: int = 40) -> str:
    """Format columns properties for display in condition name.

    Parameters
    ----------
    l : List
        list to print.
    max_elements_to_show : int , default: 10
        max elements to print before terminating.
    max_string_length : int , default: 40
        max string length before terminating.
    Returns
    -------
    str
        String of beautified list
    """
    string_list = [str(i) for i in l[:max_elements_to_show]]
    output = ', '.join(string_list)

    if len(output) > max_string_length:
        return output[:max_string_length] + '...'

    if len(l) > max_elements_to_show:
        return output + ', ...'

    return output


def format_datetime(
    value: t.Union[int, float, datetime],
) -> str:
    """Format datetime object or timestamp value.

    Parameters
    ----------
    value : Union[datetime, int, float]
        datetime (timestamp) to format
    Returns
    -------
    str
        string representation of the provided value
    Raises
    ------
    ValueError
        if unexpected value type was passed to the function
    """
    if isinstance(value, datetime):
        datetime_value = value
    elif isinstance(value, (int, float)):
        datetime_value = datetime.fromtimestamp(value)
    else:
        raise ValueError(f'Unsupported value type - {type(value).__name__}')

    if datetime_value.hour == 0 and datetime_value.minute == 0 and datetime_value.second == 0:
        return datetime_value.strftime('%Y-%m-%d')
    elif (datetime_value.hour != 0 or datetime_value.minute != 0) and datetime_value.second == 0:
        return datetime_value.strftime('%Y-%m-%d %H:%M')
    else:
        return datetime_value.strftime('%Y-%m-%d %H:%M:%S')


def create_new_file_name(file_name: str, default_suffix: str = 'html'):
    """Return file name that isn't already exist (adding (X)).

    Parameters
    ----------
    file_name : str
        the file name we want to add a (X) suffix if it exist.
    default_suffix : str , default: 'html'
        the file suffix to add if it wasn't provided in the file name.

    Returns
    -------
    str
        a new file name if the file exists
    """
    if '.' in file_name:
        basename, ext = file_name.rsplit('.', 1)
    else:
        basename = file_name
        ext = default_suffix
    file_name = f'{basename}.{ext}'
    c = itertools.count()
    next(c)
    while os.path.exists(file_name):
        file_name = f'{basename} ({str(next(c))}).{ext}'
    return file_name
