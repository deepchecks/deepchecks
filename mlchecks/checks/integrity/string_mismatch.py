"""String mismatch functions."""
from collections import defaultdict
from typing import Union, List, Set, Dict

from pandas import DataFrame, StringDtype, Series

from mlchecks import validate_dataset_or_dataframe, CheckResult
from mlchecks.checks.integrity.string_utils import string_baseform
from mlchecks.display import format_check_display

__all__ = ['string_mismatch']


def string_mismatch(dataset: DataFrame, ignore_columns: Union[str, List[str]]) -> CheckResult:
    """Detect different variants of string categories (e.g. "mislabeled" vs "mis-labeled") in a categorical column.

    Args:
        dataset (DataFrame)
        ignore_columns (Union[str, List[str]]): columns to ignore
    """
    # Validate parameters
    dataset = validate_dataset_or_dataframe(dataset)
    dataset = dataset.drop_columns_with_validation(ignore_columns)

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

    visual = df_graph.to_html() if len(df_graph) > 0 else None
    formatted_html = format_check_display('String Mismatch', string_mismatch, visual)
    return CheckResult(df_graph, display={'text/html': formatted_html})


def get_base_form_to_variants_dict(uniques):
    """Create dict of base-form of the uniques to their values.

    function gets a set of strings, and returns a dictionary of shape Dict[str]=set,
    the key being the "base_form" (a clean version of the string),
    and the value being a set of all existing original values.
    This is done using the StringCategory class.
    """
    base_form_to_variants: Dict[str, Set] = defaultdict(set)
    for item in uniques:
        base_form_to_variants[string_baseform(item)].add(item)
    return base_form_to_variants
