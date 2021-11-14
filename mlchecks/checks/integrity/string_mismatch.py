"""String mismatch functions."""
from typing import Union, Iterable

import pandas as pd

from mlchecks import CheckResult, SingleDatasetBaseCheck, Dataset, ensure_dataframe_type
from mlchecks.base.dataframe_utils import filter_columns_with_validation
from mlchecks.feature_importance_utils import calculate_feature_importance_or_null, column_importance_sorter_df
from mlchecks.string_utils import get_base_form_to_variants_dict, is_string_column, format_percent

__all__ = ['StringMismatch']


class StringMismatch(SingleDatasetBaseCheck):
    """Detect different variants of string categories (e.g. "mislabeled" vs "mis-labeled") in a categorical column."""

    def __init__(self, columns: Union[str, Iterable[str]] = None, ignore_columns: Union[str, Iterable[str]] = None,
                 n_top_columns: int = 10):
        """Initialize the StringMismatch check.

        Args:
            columns (Union[str, Iterable[str]]): Columns to check, if none are given checks all columns except ignored
                    ones.
            ignore_columns (Union[str, Iterable[str]]): Columns to ignore, if none given checks based on columns
                    variable
        n_top_columns (int): (optinal - used only if model was specified)
                             amount of columns to show ordered by feature importance (date, index, label are first)
        """
        super().__init__()
        self.columns = columns
        self.ignore_columns = ignore_columns
        self.n_top_columns = n_top_columns

    def run(self, dataset, model=None) -> CheckResult:
        """Run check.

        Args:
            dataset (DataFrame): A dataset or pd.FataFrame object.
        """
        feature_importances = calculate_feature_importance_or_null(dataset, model)
        return self._string_mismatch(dataset, feature_importances)

    def _string_mismatch(self, dataset: Union[pd.DataFrame, Dataset],
                         feature_importances: pd.Series=None) -> CheckResult:
        # Validate parameters
        original_dataset = dataset
        dataset: pd.DataFrame = ensure_dataframe_type(dataset)
        dataset = filter_columns_with_validation(dataset, self.columns, self.ignore_columns)

        results = []

        for column_name in dataset.columns:
            column: pd.Series = dataset[column_name]
            if not is_string_column(column):
                continue

            uniques = column.unique()
            base_form_to_variants = get_base_form_to_variants_dict(uniques)
            for base_form, variants in base_form_to_variants.items():
                if len(variants) == 1:
                    continue
                for variant in variants:
                    count = sum(column == variant)
                    results.append([column_name, base_form, variant, count, format_percent(count / dataset.size)])

        # Create dataframe to display graph
        df_graph = pd.DataFrame(results, columns=['Column Name', 'Base form', 'Value', 'Count', '% In data'])
        df_graph = df_graph.set_index(['Column Name', 'Base form'])
        df_graph = column_importance_sorter_df(df_graph, original_dataset, feature_importances,
                                               self.n_top_columns, col='Column Name')

        if len(df_graph) > 0:
            display = df_graph
        else:
            display = None

        return CheckResult(df_graph, check=self.__class__, display=display)
