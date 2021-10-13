"""The single_feature_contribution check module."""
import re
from copy import deepcopy
from typing import Union, List

import pandas as pd

from mlchecks import CheckResult, Dataset, SingleDatasetBaseCheck
from mlchecks.base.dataset import validate_dataset_or_dataframe
from mlchecks.display import format_check_display

__all__ = ['rare_format_detection', 'RareFormatDetection', 'SubStr', 'Pattern']


class SubStr:
    """Supporting class for regex subbing in pattern."""

    def __init__(self, regex_str: str, filler: str):
        """Initiate the SubStr class."""
        self.regex_str = regex_str
        self.filler = filler

    def sub(self, s: str) -> str:
        """Replace matching patterns to the regex_str by the filler."""
        return re.sub(self.regex_str, self.filler, s)


class Pattern:
    """Supporting class for creating complicated patterns for rare_format_detection."""

    def __init__(self, name: str, substrs: List[SubStr]):
        """Initiate the Pattern class."""
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
                          patterns: List[Pattern] = deepcopy(DEFAULT_PATTERNS), rarity_threshold: float = 0.05) \
        -> CheckResult:
    """
    Check whether columns have common formats (e.g. "XX-XX-XXXX" for dates") and detects values that don't match.

    Args:
        dataset (Dataset): A dataset object
        column_names: list of columns or name of column to run on. Uses all feature columns if not specified.
        patterns: patterns to look for when comparing common vs. rare formats. Uses DEFAULT_PATTERNS if not specified
        rarity_threshold: threshold for get_rare_vs_common_values function

    Returns:
        CheckResult:
            - value: dictionary of all columns and found patterns
            - display: pandas Dataframe per column, showing the rare-to-common-ratio, common formats, examples for
                       common values and rare values
    """
    dataset = validate_dataset_or_dataframe(dataset)
    column_names = column_names or dataset.features()

    if isinstance(column_names, str):
        column_names = [column_names]

    res = {column_name: _detect_per_column(dataset[column_name], patterns, rarity_threshold)
           for column_name in column_names}

    html = ''
    for key, value in res.items():
        if value is not None and not value.empty:
            html += f'Column {key}:<br><br>{value.to_html()}<br>'

    return CheckResult(res, {'text/html': format_check_display('Rare Format Detection', rare_format_detection, html)})


def _detect_per_column(column: pd.Series, patterns, rarity_threshold):
    """
    Check whether a column has common formats (e.g. "XX-XX-XXXX" for dates") and detects values that don't match.

    Args:
        column: A pandas Series object
        patterns: patterns to look for when comparing common vs. rare formats
        rarity_threshold: threshold for get_rare_vs_common_values function

    Returns:
        pandas Dataframe: table showing the rare-to-common-ratio, common formats, examples for common values and
                          rare values
    """
    all_pattern_janus_results = {pattern.name: _detect_per_column_and_pattern(column, pattern, rarity_threshold)
                                 for pattern in patterns}

    return pd.DataFrame(all_pattern_janus_results).dropna(axis=1, how='all')


def _detect_per_column_and_pattern(column, pattern, rarity_threshold):
    """
    Check whether a column has common formats (e.g. "XX-XX-XXXX" for dates") and detects values that don't match.

    This function checks one pattern per column.

    Args:
        column: A pandas Series object
        pattern: pattern to look for when comparing common vs. rare formats
        rarity_threshold: threshold for get_rare_vs_common_values function

    Returns:
        dict: dictionary with values representing the rare-to-common-ratio, common formats, examples for common values
              and rare values
    """
    column = column.astype(str)

    # find rare formats by replacing every pattern with the filler.
    # For example if the pattern is \d+ and column contains dates, the formats that will be found is _._._
    def replace_pattern(s: str, pattern):
        for substr in pattern.substrs:
            s = substr.sub(s)
        return s

    patterned_column = column.apply(lambda s: replace_pattern(s, pattern))

    rare_to_common_format_ratio, rare_formats, common_formats = get_rare_vs_common_values(patterned_column,
                                                                                          rarity_threshold)

    if rare_to_common_format_ratio == 0: return {}

    rare_values = column[patterned_column.isin(rare_formats)]
    common_values_examples = [column[patterned_column == common_format].values[1] for common_format in common_formats]

    return {'ratio of rare patterns to common patterns': f'{rare_to_common_format_ratio:.2%}',
            'common formats': common_formats,
            'examples for values in common formats': common_values_examples,
            'values in rare formats': list(rare_values.unique())}


def get_rare_vs_common_values(col: pd.Series, sharp_drop_ratio_threshold: float = 0.05):
    """
    Look for a sudden drop in prevalence of values, and returns ratio of rare to common values and the actual values.

    The function defines which values are rare or common by the following logic:
    For each value, by descending order of commonness, we check how common this value is compared to the previous
    (more common) value. if there's a sudden drop (e.g. from 100 samples of previous value to 1), we consider this and
    all subsequent values to be "rare", and the previous ones to be "common".

    Args:
        col: pandas Series to check for rare values
        sharp_drop_ratio_threshold: threshold under which values are considered "rare", as described above.

    Returns:
        rare_to_common_format_ratio (float): ratio of all rare samples to common samples
        rare_values (list): list of rare values
        common_values (list): list of common values

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
    rare_values = value_counts.iloc[i + 1:]  # pylint: disable=undefined-loop-variable
    common_values = value_counts.iloc[:i + 1]  # pylint: disable=undefined-loop-variable

    rare_to_common_format_ratio = rare_values.sum() / common_values.sum()

    return rare_to_common_format_ratio, list(rare_values.index), list(common_values.index)


class RareFormatDetection(SingleDatasetBaseCheck):
    """Checks whether columns have common formats (e.g. "XX-XX-XXXX" for dates") and detects values that don't match."""

    def run(self, dataset: Dataset, model=None) -> CheckResult:
        """
        Run the rare_format_detection function.

        Args:
            dataset: Dataset - The dataset object
            model: any = None - not used in the check

        Returns:
            CheckResult:
                - value: dictionary of all columns and found patterns
                - display: pandas Dataframe per column, showing the rare-to-common-ratio, common formats, examples for
                           common values and rare values
        """
        return rare_format_detection(dataset=dataset, rarity_threshold=self.params.get('rarity_threshold', 0.05))
