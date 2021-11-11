"""The single_feature_contribution check module."""
import re
from copy import deepcopy
from typing import Iterable, Union, List, Tuple

import pandas as pd

from mlchecks import CheckResult, Dataset, SingleDatasetBaseCheck
from mlchecks.base.dataframe_utils import filter_columns_with_validation
from mlchecks.base.dataset import ensure_dataframe_type
from mlchecks.string_utils import split_and_keep, split_by_order, format_percent
from mlchecks.utils import MLChecksValueError

__all__ = ['RareFormatDetection', 'Pattern']


class Pattern:
    """Supporting class for creating complicated patterns for rare_format_detection."""

    def __init__(self, name: str, substituters: Union[List[Tuple[str, str]], Tuple[str, str]], ignore: str = None,
                 refine: bool = False, is_sequence: bool = False):
        """
        Initiate the Pattern class.

        Args:
            name: name of pattern, will be shown in the results.
            substituters: list of tuples or just a tuple. first argument in the tuple is the regex string, to find
                relevant patterns. second argument in the tuple is a substring to replace all relevant substrings.
            ignore: regex string indicating which substrings should be ignored (replaced with '')
            refine: boolean. Indicates whether this pattern should be refined later (see _refine_formats)
            is_sequence: boolean. Indicates whether the substituters are for one characters or for a sequence. Relevant
                only when refine is True.
        """
        self.name = name
        if isinstance(substituters, tuple):
            substituters = [substituters]
        self.substituters = substituters  # tuple or list of tuples of regex and str
        self.ignore = ignore  # regex
        self.refine = refine
        self.is_sequence = is_sequence

    def sub(self, s: str) -> str:
        """Replace matching patterns to the regex_str by the filler."""
        s = self.clean(s)
        for subs in self.substituters:
            s = re.sub(subs[0], subs[1], s)
        return s

    def clean(self, s: str) -> str:
        """Remove all substrings that should be ignored."""
        if self.ignore is None:
            return s
        return re.sub(self.ignore, '', s)

    def is_format_significant(self, fmt) -> bool:
        """Return boolean indicating whether format includes any filler."""
        is_substr_in_format = any(sub[1] in fmt for sub in self.substituters)
        is_format_more_than_just_repeating_substr = fmt.count(fmt[0]) != len(fmt)
        return is_substr_in_format and is_format_more_than_just_repeating_substr


DEFAULT_PATTERNS = [
    Pattern(name='digits and letters format (case sensitive)',
            substituters=[(r'\d', '0'), (r'[A-Z]', 'X'), (r'[a-z]', 'x')], refine=True),
    Pattern(name='digits and letters format', substituters=[(r'\d', '0'), (r'[A-Z|a-z]', 'X')], refine=True),
    Pattern(name='digits only format (ignoring letters)', substituters=(r'\d', '0'), ignore=r'[A-Z|a-z]',
            refine=True),
    Pattern(name='letters only format (ignoring digits)', substituters=(r'[A-Z|a-z]', 'X'), ignore=r'\d',
            refine=True),
    Pattern(name='digits or letters format', substituters=(r'[A-Z|a-z|d]', 'X'), refine=True, is_sequence=True),
    Pattern(name='sequences of digits only format (ignoring letters)', substituters=(r'\d+', '000'),
            ignore=r'[A-Z|a-z]', refine=True, is_sequence=True),
    Pattern(name='sequences of letters only format (ignoring letters)', ignore=r'\d',
            substituters=(r'[A-Z|a-z]+', 'XXX'), refine=True, is_sequence=True),
    Pattern(name='any sequence format', substituters=(r'[A-Z|a-z|d]+', 'XXX'), refine=True, is_sequence=True),
]


def _detect_per_column(column: pd.Series, patterns: List[Pattern], rarity_threshold: float,
                       min_unique_common_ratio: float, pattern_match_method: str) -> pd.DataFrame:
    """
    Check whether a column has common formats (e.g. "XX-XX-XXXX" for dates") and detects values that don't match.

    Args:
        column (pd.Series): A pandas Series object
        patterns (List[Pattern]): patterns to look for when comparing common vs. rare formats
        rarity_threshold (float): threshold for get_rare_vs_common_values function
        min_unique_common_ratio (float): minimum ratio for unique common samples to all common samples
        pattern_match_method (str): 'first' or 'all'. If 'first', returns only the pattern where a "rare format" sample
            was found for the first time. If 'all', returns all patterns in which anything was found.


    Returns:
        pandas Dataframe: table showing the rare-to-common-ratio, common formats, examples for common values and
                          rare values
    """
    # all_pattern_results = {pattern.name: _detect_per_column_and_pattern(column, pattern, rarity_threshold)
    #                        for pattern in patterns}
    all_pattern_results = {}
    formats_to_ignore = []
    for pattern in patterns:
        res = _detect_per_column_and_pattern(column, pattern, rarity_threshold, min_unique_common_ratio,
                                             formats_to_ignore)
        if res and pattern_match_method == 'first':
            formats_to_ignore.extend(res['values in rare formats'])
        all_pattern_results[pattern.name] = res

    return pd.DataFrame(all_pattern_results).dropna(axis=1, how='all')


def _detect_per_column_and_pattern(column: pd.Series, pattern: Pattern, rarity_threshold: float,
                                   min_unique_common_ratio: float, exclude_samples: list = None) -> dict:
    """
    Check whether a column has common formats (e.g. "XX-XX-XXXX" for dates") and detects values that don't match.

    This function checks one pattern per column.

    Args:
        column (pd.Series): A pandas Series object
        pattern (Pattern): pattern to look for when comparing common vs. rare formats
        rarity_threshold (float): threshold for get_rare_vs_common_values function
        min_unique_common_ratio (float): minimum ratio for unique common samples to all common samples
        exclude_samples (List[str]): list of samples to ignore

    Returns:
        dict: dictionary with values representing the rare-to-common-ratio, common formats, examples for common values
              and rare values
    """
    column = column.astype(str)

    # find rare formats by replacing every pattern with the filler.
    # For example if the pattern is \d+ and column contains dates, the formats that will be found is _._._
    patterned_column = column.apply(pattern.sub)

    rare_formats, common_formats = get_rare_vs_common_values(patterned_column, rarity_threshold)

    if len(rare_formats) == 0: return {}
    if not any(pattern.is_format_significant(common_format) for common_format in common_formats): return {}

    rare_values = column[patterned_column.isin(rare_formats)]
    rare_values = rare_values[~rare_values.isin(exclude_samples)]
    if rare_values.empty: return {}

    num_rare_formats = rare_values.shape[0]
    rare_format_ratio = num_rare_formats / column.shape[0]

    common_values = column[patterned_column.isin(common_formats)]
    if common_values.nunique() / common_values.shape[0] < min_unique_common_ratio: return {}

    common_values_examples = [column[patterned_column == common_format].values[0] for common_format in common_formats]

    if pattern.refine is True:
        for i, fmt in enumerate(common_formats):
            format_samples = column[patterned_column == fmt].apply(pattern.clean).values
            common_formats[i] = _refine_formats(
                fmt=fmt, substrs=[sub[1] for sub in pattern.substituters], samples=format_samples,
                is_substr_sequence=pattern.is_sequence
            )

    return {'ratio of rare samples': f'{format_percent(rare_format_ratio)} ({num_rare_formats})',
            'common formats': common_formats,
            'examples for values in common formats': common_values_examples,
            'values in rare formats': list(rare_values.unique())}


def _refine_formats(fmt: str, substrs: List[str], samples: List[str], is_substr_sequence: bool = False) -> str:
    """
    Return a refined (degeneralized) pattern, based on known samples.

    Example: format "XXX@XXX.XXX" was found. However, it appears that all samples of this format are in gmail.com
    domain. This function detects that and returns a degeneralized format - "XXX@gmail.com"

    Args:
        fmt (str): string representing the format
        substrs (str): original substr of the format that replaced other characters when detecting the pattern
        samples (List[str]): list of all known samples of current format
        is_substr_sequence (bool): boolean representing whether the substr replaced each character (e.g. '/d') or
                                   a whole sequence (e.g. '/d+')

    Returns:
        str: degeneralized format, based on the fmt input
    """
    if all(substr not in fmt for substr in substrs):
        return fmt

    if is_substr_sequence is True:
        splt_fmt = split_and_keep(s=fmt, separators=substrs)
        splt_fmt_wo_sep = list(filter(lambda x: x not in substrs, splt_fmt))
        split_examples = [split_by_order(s, splt_fmt_wo_sep) for s in samples]
    else:
        splt_fmt = list(fmt)
        split_examples = [list(s) for s in samples]

    new_format = []
    example = split_examples[0]
    for i in range(len(example)):
        if splt_fmt[i] not in substrs:
            new_format.append(splt_fmt[i])
            continue

        common_value = example[i]
        # If separator represents the same string in all examples, use that instead of a separator:
        if all(split_sample[i] == common_value for split_sample in split_examples[1:]):
            new_format.append(common_value)
        else:
            new_format.append(splt_fmt[i])

    new_format = ''.join(new_format)

    return new_format


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
        rare_values (list): list of rare values
        common_values (list): list of common values

    """
    # should do: analyze numeric data differently - consider analyze the range of the numbers in cols and detect
    #  rare values that are out of this range

    # apply the func over the col
    value_counts = col.value_counts().drop('', errors='ignore')  # Ignoring patterns that result in empty string

    # if there is only one value in the value count, there are no rare values
    if len(value_counts) <= 1: return [], []

    # look for a sharp decline (at least sharp_drop_ratio_threshold)
    for i, (prev_value, curr_value) in enumerate(zip(value_counts[:-1], value_counts[1:])):
        if curr_value < sharp_drop_ratio_threshold * prev_value:
            break
    else:
        return [], []

    # returns the rare values and their portion compared to the most common value
    rare_values = value_counts.iloc[i + 1:]  # pylint: disable=undefined-loop-variable
    common_values = value_counts.iloc[:i + 1]  # pylint: disable=undefined-loop-variable

    return list(rare_values.index), list(common_values.index)


class RareFormatDetection(SingleDatasetBaseCheck):
    """Checks whether columns have common formats (e.g. "XX-XX-XXXX" for dates") and detects values that don't match.

    Example for a Pattern:
        Pattern(name='digits or letters format', substituters=(r'[A-Z|a-z|d]', 'X'))
        This pattern looks for either digits or letters and replaces them with the character 'X'. By replacing
        these, we can find all strings matching this certain pattern and see how common (or rare) it is.

        In this example, the string "nir123@deepchecks.com" would be changed to "XXXXXX@XXXXXXXXXX.XXX".
        All other strings matching this format (e.g. "noam12@deepchecks.com") would be identified as having the
        same pattern.

        If we also mark "refine = True" in the Pattern class, the check will further try and make the pattern
        more accurate, by trying to find common characters in all samples of the same pattern. In this example,
        the refined format found would be "XXXXXX@deepchecks.com.
    """

    def __init__(self, columns: Union[str, Iterable[str]] = None, ignore_columns: Union[str, Iterable[str]] = None,
                 patterns: List[Pattern] = deepcopy(DEFAULT_PATTERNS), rarity_threshold: float = 0.05,
                 min_unique_common_ratio=0.01, pattern_match_method: str = 'first'):
        """Initialize the RareFormatDetection check.

        Args:
            columns (Union[str, Iterable[str]]): Columns to check, if none are given checks all columns except ignored
                ones.
            ignore_columns (Union[str, Iterable[str]]): Columns to ignore, if none given checks based on columns
                variable
            patterns (List[Pattern]): patterns to look for when comparing common vs. rare formats. Uses DEFAULT_PATTERNS
                if not specified.
                Note that if pattern_match_method='first' (which it is by default), then the order of patterns matter.
                In this case, it is advised to order the patterns from specific to general.
            rarity_threshold (float): threshold to indicate what is considered a "sharp" drop in commonness of values.
                This is used by the function get_rare_vs_common_values which divides data into "common" and "rare"
                values, and is used here to determine which formats are common and which are rare.
            min_unique_common_ratio (float): minimum ratio for unique common samples to all common samples.
                This parameter is used in order to filter unwanted results in the case where the common format is
                actually a common value.
                This is because if a common format has too few unique values, it's probably actually just a categorical
                feature with some values that are very common and some that are rare.
            pattern_match_method (str): 'first' or 'all'. If 'first', returns only the pattern where a "rare format"
                sample was found for the first time. If 'all', returns all patterns in which anything was found.

        """
        super().__init__()
        self.columns = columns
        self.ignore_columns = ignore_columns
        self.patterns = patterns
        self.rarity_threshold = rarity_threshold
        self.min_unique_common_ratio = min_unique_common_ratio
        self.pattern_match_method = pattern_match_method

    def run(self, dataset: Dataset, model=None) -> CheckResult:
        """Run check.

        Args:
            dataset: Dataset - The dataset object

        Returns:
            CheckResult:
                - value: dictionary of all columns and found patterns
                - display: pandas Dataframe per column, showing the rare-to-common-ratio, common formats, examples for
                           common values and rare values
        """
        return self._rare_format_detection(dataset=dataset)

    def _rare_format_detection(self, dataset: Union[Dataset, pd.DataFrame]) -> CheckResult:
        dataset: pd.DataFrame = ensure_dataframe_type(dataset)
        dataset = filter_columns_with_validation(dataset, self.columns, self.ignore_columns)

        if self.pattern_match_method not in ['first', 'all']:
            raise MLChecksValueError(f'pattern_match_method must be "first" or "all", got {self.pattern_match_method}')

        res = {
            column_name: _detect_per_column(dataset[column_name].dropna(), self.patterns, self.rarity_threshold,
                                            self.min_unique_common_ratio, self.pattern_match_method)
            for column_name in dataset.columns}

        display = []
        for key, value in res.items():
            if value.shape[0] > 0:
                display.append(f'\n\nColumn {key}:')
                display.append(value)

        return CheckResult(value=res, header='Rare Format Detection', check=self.__class__, display=display)
