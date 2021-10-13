"""The single_feature_contribution check module."""
import re
from typing import Union, List

import pandas as pd

from mlchecks import CheckResult, Dataset, SingleDatasetBaseCheck
from mlchecks.base.dataset import validate_dataset_or_dataframe
from mlchecks.display import format_check_display
from mlchecks.utils import MLChecksValueError

__all__ = ['rare_format_detection', 'RareFormatDetection', 'SubStr', 'Pattern']


class SubStr:

    def __init__(self, regex_str: str, filler: str):
        self.regex_str = regex_str
        self.filler = filler

    def sub(self, s: str) -> str:
        return re.sub(self.regex_str, self.filler, s)


class Pattern:

    def __init__(self, name: str, substrs: List[SubStr]):
        self.name = name
        self.substrs = substrs


DEFAULT_PATTERNS = [
    Pattern('digits only format (ignoring letters)', [SubStr(r'\d', '0'), SubStr(r'[A-Z|a-z]', '')]),
    Pattern('sequences of digits only format (ignoring letters)', [SubStr(r'\d', 'DIGIT_SEQ'),
                                                                   SubStr(r'[A-Z|a-z]', '')]),
    Pattern('letters only format (ignoring digits)', [SubStr(r'\d', ''), SubStr(r'[A-Z|a-z]', '')]),
    Pattern('sequences of letters only format (ignoring letters)', [SubStr(r'\d', ''),
                                                                    SubStr(r'[A-Z|a-z]+', 'LETTER_SEQ')]),
    Pattern('digits and letters format', [SubStr(r'\d', '0'), SubStr(r'[A-Z|a-z]', 'X')]),
    Pattern('digits and letters format (case sensitive)', [SubStr(r'\d', '0'), SubStr(r'[A-Z]', 'X'),
                                                           SubStr(r'[a-z]', 'x')]),
    Pattern('any sequence format', [SubStr(r'\d', '0'), SubStr(r'[A-Z|a-z|d]+', 'SEQ')]),
]


def rare_format_detection(dataset: Union[Dataset, pd.DataFrame], column_names: Union[str, List[str]] = None,
                          patterns: List[Pattern] = DEFAULT_PATTERNS, rarity_threshold: float = 0.05) -> CheckResult:
    """
    Checks whether columns have common formats (e.g. "XX-XX-XXXX" for dates") and detects values that don't match.

    Args:
        dataset:
        column_names:
        patterns:
        rarity_threshold:

    Returns:

    """

    dataset = validate_dataset_or_dataframe(dataset)
    column_names = column_names or dataset.features()

    if isinstance(column_names, str):
        res = {column_names: _string_format_validation_column(dataset[column_names], patterns, rarity_threshold)}
    elif isinstance(column_names, list):
        res = {column_name: _string_format_validation_column(dataset[column_name], patterns, rarity_threshold)
               for column_name in column_names}
    else:
        raise MLChecksValueError(f'column_names must be either list or str. instead got {type(column_names).__name__}')

    html = ''
    for key, value in res.items():
        if value is not None and not value.empty:
            html += f'Column {key}:<br><br>{value.to_html()}<br>'

    return CheckResult(res, {'text/html': format_check_display('rare_format_detection', rare_format_detection, html)})


def _string_format_validation_column(column: pd.Series, patterns, rarity_threshold):
    all_pattern_janus_results = {pattern.name: _check_formats_and_values(column, pattern, rarity_threshold)
                                 for pattern in patterns}

    return pd.DataFrame(all_pattern_janus_results).dropna(axis=1, how='all')


def _check_formats_and_values(column, pattern, rarity_threshold):
    column = column.astype(str)

    # find rare formats by replacing every pattern with the filler.
    # For example if the pattern is \d+ and column contains dates, the formats that will be found is _._._
    def replace_pattern(s: str, pattern):
        for substr in pattern.substrs:
            s = substr.sub(s)
        return s

    patterned_column = column.apply(lambda s: replace_pattern(s, pattern))

    rare_to_common_format_ratio, rare_formats, common_formats = detect_rare(patterned_column, rarity_threshold)

    if rare_to_common_format_ratio == 0: return {}

    rare_values = column[patterned_column.isin(rare_formats)]
    common_values_examples = [column[patterned_column == common_format].values[1] for common_format in common_formats]

    return {'ratio of rare patterns to common patterns': f'{rare_to_common_format_ratio:.2%}',
            'common formats': common_formats,
            'examples for values in common formats': common_values_examples,
            'values in rare formats': list(rare_values.unique())}


def detect_rare(col: pd.Series, sharp_drop_ratio_threshold: float = 0.05):
    """
    apply the function over the col and look for rare values in the result.
    To detect rare values, the function runs a value_count over the result, and look for a count that is at least
    MINIMAL_RATIO_BETWEEN_OUT_OF_FORMAT_TO_FORMAT_PORTIONS smaller than the previous count.
    If such exist, any value that has this count or smaller will be count as rare.
    """

    # should do: analyze numeric data differently - consider analyze the range of the numbers in cols and detect
    #  rare values that are out of this range

    # apply the func over the col
    value_counts = col.value_counts().drop('', errors='ignore')  # Ignoring patterns that result in empty string

    # if there is only one value in the value count, there are no rare values
    if len(value_counts) <= 1: return 0, None, None

    # look for a sharp decline (at least sharp_drop_ratio_threshold)
    for i, (prev_value, curr_value) in enumerate(zip(value_counts[:-1], value_counts[1:])):
        if curr_value < sharp_drop_ratio_threshold * prev_value:
            break
    else:
        return 0, None, None

    # returns the rare values and their portion compared to the most common value
    rare_values = value_counts.iloc[i + 1:]
    common_values = value_counts.iloc[:i + 1]

    rare_to_common_format_ratio = rare_values.sum() / common_values.sum()

    return rare_to_common_format_ratio, list(rare_values.index), list(common_values.index)


class RareFormatDetection(SingleDatasetBaseCheck):

    def run(self, dataset: Dataset, model=None) -> CheckResult:
        return rare_format_detection(dataset=dataset, rarity_threshold=self.params.get('rarity_threshold', 0.05))
