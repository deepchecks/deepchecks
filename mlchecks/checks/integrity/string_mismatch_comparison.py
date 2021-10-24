"""String mismatch functions."""
from typing import Union, Iterable

import pandas as pd
from pandas import DataFrame, StringDtype, Series

from mlchecks import CheckResult, Dataset, ensure_dataframe_type, CompareDatasetsBaseCheck
from mlchecks.base.dataframe_utils import filter_columns_with_validation
from mlchecks.base.string_utils import get_base_form_to_variants_dict

__all__ = ['string_mismatch_comparison', 'StringMismatchComparison']


def percentage_in_series(series, values):
    count = sum(series.isin(values))
    return f'{count} | {count / series.size:.2%}'


def string_mismatch_comparison(dataset: Union[pd.DataFrame, Dataset],
                               compared_dataset: Union[pd.DataFrame, Dataset],
                               columns: Union[str, Iterable[str]] = None,
                               ignore_columns: Union[str, Iterable[str]] = None) -> CheckResult:
    """Detect different variants of string categories between the same categorical column in two datasets.

    This check compares the same categorical column within 2 different datasets and checks whether
    there are variants of similar strings that do not exist in both.
    Specifically, we define similarity between strings if they are equal when ignoring case and non-letter characters.
    Example:
    We have a dataset with similar strings 'string' and 'St. Ring', which have different meanings.
    Our compared data has the strings 'string', 'St. Ring' and a new phrase, 'st.  ring'.
    Here, we have a new variant of the above strings, and would like to be acknowledged, as this is obviously a
    different version of 'St. Ring'.


    Args:
        dataset (Union[pd.DataFrame, Dataset]): A dataset or pd.FataFrame object.
        compared_dataset (Union[pd.DataFrame, Dataset]): A dataset or pd.FataFrame object.
        columns (Union[str, Iterable[str]]): Columns to check, if none are given checks all columns except ignored ones.
        ignore_columns (Union[str, Iterable[str]]): Columns to ignore, if none given checks based on columns variable
    """
    # Validate parameters
    df: pd.DataFrame = ensure_dataframe_type(dataset)
    df = filter_columns_with_validation(df, columns, ignore_columns)
    compared_df: pd.DataFrame = ensure_dataframe_type(compared_dataset)

    mismatches = []

    # Get shared columns
    columns = set(df.columns).intersection(compared_df.columns)

    for column_name in columns:
        tested_column: Series = dataset[column_name].convert_dtypes()
        compared_column: Series = compared_df[column_name].convert_dtypes()
        # If one of the columns isn't string type, continue
        if not StringDtype.is_dtype(tested_column.dtype) or not StringDtype.is_dtype(compared_column.dtype):
            continue

        tested_baseforms = get_base_form_to_variants_dict(tested_column.unique())
        compared_baseforms = get_base_form_to_variants_dict(compared_column.unique())

        common_baseforms = set(tested_baseforms.keys()).intersection(compared_baseforms.keys())
        for baseform in common_baseforms:
            tested_values = tested_baseforms[baseform]
            compared_values = compared_baseforms[baseform]
            # If at least one value is unique to either of the datasets, add the column to results
            if tested_values.symmetric_difference(compared_values):
                # Calculate all values to be shown
                variants_only_in_dataset = tested_values - compared_values
                variants_only_in_compared = compared_values - tested_values
                common_variants = tested_values & compared_values
                percent_variants_only_in_dataset = percentage_in_series(tested_column, variants_only_in_dataset)
                percent_variants_in_compared = percentage_in_series(compared_column, variants_only_in_compared)

                mismatches.append([column_name, baseform, variants_only_in_dataset, variants_only_in_compared,
                                   common_variants, percent_variants_only_in_dataset, percent_variants_in_compared])

    # Create result dataframe
    df_graph = DataFrame(mismatches,
                         columns=['Column name', 'Base form', 'Variants only in dataset',
                                  'Variants only  in comparison dataset', 'Common variants',
                                  '% Variants only in dataset', '% Variants only  in comparison dataset'])
    df_graph = df_graph.set_index(['Column name', 'Base form'])
    # For display transpose the dataframe
    display = df_graph.T if len(df_graph) > 0 else None

    return CheckResult(df_graph, check=string_mismatch_comparison, display=display)


class StringMismatchComparison(CompareDatasetsBaseCheck):
    """Detect different variants of string categories between the same categorical column in two datasets."""

    def run(self, dataset, compared_dataset, model=None) -> CheckResult:
        """Run string_mismatch_comparison check.

        Args:
            dataset (Dataset): A dataset object.
            compared_dataset (Dataset): A dataset object.
            model: Not used in this check.
        """
        return string_mismatch_comparison(dataset,
                                          compared_dataset,
                                          columns=self.params.get('columns'),
                                          ignore_columns=self.params.get('ignore_columns'))
