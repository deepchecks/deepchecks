"""module contains Data Duplicates check."""
from typing import Union, Iterable

import pandas as pd

from deepchecks import Dataset, ensure_dataframe_type
from deepchecks.base.check import CheckResult, SingleDatasetBaseCheck
from deepchecks.base.dataframe_utils import filter_columns_with_validation
from deepchecks.utils import DeepchecksValueError
from deepchecks.string_utils import format_percent


__all__ = ['LabelAmbiguity']


class LabelAmbiguity(SingleDatasetBaseCheck):
    """Search for label ambiguity in dataset."""

    def __init__(self, columns: Union[str, Iterable[str]] = None, ignore_columns: Union[str, Iterable[str]] = None,
                 n_to_show: int = 5):
        """Initialize the LabelAmbiguity class.

        Args:
            columns (str, Iterable[str]): List of columns to check, if none given checks all columns Except ignored
            ones.
            ignore_columns (str, Iterable[str]): List of columns to ignore, if none given checks based on columns
            variable.
            n_to_show (int): number of most common ambiguous samples to show.

        """
        super().__init__()
        self.columns = columns
        self.ignore_columns = ignore_columns
        self.n_to_show = n_to_show

    def run(self, dataset: Dataset, model=None) -> CheckResult:
        """Run check.

        Args:
          dataset(Dataset): any dataset.

        Returns:
          (CheckResult): percentage of ambiguous samples and display of the top n_to_show most ambiguous.
        """
        dataset: Dataset = Dataset.validate_dataset(dataset)
        dataset = dataset.filter_columns_with_validation(self.columns, self.ignore_columns)

        if dataset.n_samples() == 0:
            raise DeepchecksValueError('Dataset does not contain any data')

        group_unique_data = dataset.data.groupby(dataset.features(), dropna=False)
        group_unique_labels = group_unique_data.nunique()[dataset.label_name()]
        num_ambiguous = 0
        display = None

        for num_labels, group_data in sorted(zip(group_unique_labels, group_unique_data),
                                             key=lambda x: x[0], reverse=True):
            if num_labels == 1:
                break

            group_df = group_data[1]
            label_counts = dict(group_df.groupby('c').size())
            n_data_sample = group_df.shape[0]
            num_ambiguous += n_data_sample

            #Todo: turn label_counts into display
            '''
            should be something like:
            count, label, data
            1    , 1    , fasdflasdkjf;asdjf
            7    , 0    , fasdflasdkjf;asdjf
            10   , 1    , other_data
            10   , 0    , other_data
            '''


        percent_ambiguous = num_ambiguous/dataset.n_samples()

        return CheckResult(value=percent_ambiguous, check=self.__class__, display=display)
