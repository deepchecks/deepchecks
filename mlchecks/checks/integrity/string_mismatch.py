"""String mismatch functions."""
from typing import Union, Iterable

import pandas as pd
from pandas import DataFrame, StringDtype, Series

from mlchecks import CheckResult, SingleDatasetBaseCheck, Dataset, ensure_dataframe_type
from mlchecks.base.dataframe_utils import filter_columns_with_validation
from mlchecks.base.string_utils import get_base_form_to_variants_dict

__all__ = ['string_mismatch', 'StringMismatch']


def string_mismatch(dataset: Union[pd.DataFrame, Dataset], columns: Union[str, Iterable[str]] = None,
                    ignore_columns: Union[str, Iterable[str]] = None) -> CheckResult:
    """Detect different variants of string categories (e.g. "mislabeled" vs "mis-labeled") in a categorical column.

    Args:
        dataset (DataFrame): A dataset or pd.FataFrame object.
        columns (Union[str, Iterable[str]]): Columns to check, if none are given checks all columns except ignored ones.
        ignore_columns (Union[str, Iterable[str]]): Columns to ignore, if none given checks based on columns variable
    """
    # Validate parameters
    dataset: pd.DataFrame = ensure_dataframe_type(dataset)
    dataset = filter_columns_with_validation(dataset, columns, ignore_columns)

    results = []

    for column_name in dataset.columns:
        column: Series = dataset[column_name]
        # TODO: change to if check column is categorical
        if column.dtype != StringDtype:
            continue

        uniques = column.unique()
        base_form_to_variants = get_base_form_to_variants_dict(uniques)
        for base_form, variants in base_form_to_variants.items():
            if len(variants) == 1:
                continue
            for variant in variants:
                count = sum(column == variant)
                results.append([column_name, base_form, variant, count, round(count / dataset.size, 2)])

    # Create dataframe to display graph
    df_graph = DataFrame(results, columns=['Column Name', 'Base form', 'Value', 'Count', 'Fraction of data'])
    df_graph = df_graph.set_index(['Column Name', 'Base form'])

    if len(df_graph) > 0:
        display = df_graph
    else:
        display = None

    return CheckResult(df_graph, check=string_mismatch, display=display)


class StringMismatch(SingleDatasetBaseCheck):
    """Detect different variants of string categories (e.g. "mislabeled" vs "mis-labeled") in a categorical column."""

    def run(self, dataset, model=None) -> CheckResult:
        """Run string_mismatch check.

        Args:
            dataset (DataFrame): A dataset or pd.FataFrame object.
        """
        return string_mismatch(dataset,
                               columns=self.params.get('columns'),
                               ignore_columns=self.params.get('ignore_columns'))
