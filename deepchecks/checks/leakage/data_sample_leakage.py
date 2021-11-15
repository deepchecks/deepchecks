"""The data_sample_leakage_report check module."""
from typing import Dict, List
import re

from deepchecks import Dataset
from deepchecks.base.check import CheckResult, TrainValidationBaseCheck
from deepchecks.string_utils import format_percent

import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None

__all__ = ['DataSampleLeakageReport']


def get_dup_indexes_map(df: pd.DataFrame, columns: List) -> Dict:
    """Find duplicated indexes in the dataframe.

    Args:
        df: a Dataframe object of the dataset
        columns: list of column that duplicates are defined by
    Returns:
        dictionary of each of the first indexes and its' duplicated indexes

    """
    dup = df[df.duplicated(columns, keep=False)].groupby(columns).groups.values()
    dup_map = {}
    for i_arr in dup:
        key = i_arr[0]
        dup_map[key] = [int(i) for i in i_arr[1:]]
    return dup_map


def get_dup_txt(i: int, dup_map: Dict) -> str:
    """Return a prettified text for a key in the dict.

    Args:
        i: the index key
        dup_map: the dict of the duplicated indexes
    Returns:
        prettified text for a key in the dict

    """
    val = dup_map.get(i)
    if not val:
        return str(i)
    txt = f'{i}, '
    for j in val:
        txt += f'{j}, '
    txt = txt[:-2]
    if len(txt) < 30:
        return txt
    return f'{txt[:30]}.. Tot. {(1 + len(val))}'


class DataSampleLeakageReport(TrainValidationBaseCheck):
    """Detect samples in the validation data that appear also in training data."""

    def run(self, train_dataset: Dataset, validation_dataset: Dataset,  model=None) -> CheckResult:
        """Run check.

        Args:
            train_dataset (Dataset): The training dataset object. Must contain an index.
            validation_dataset (Dataset): The validation dataset object. Must contain an index.
            model (): any = None - not used in the check
        Returns:
            CheckResult: value is sample leakage ratio in %,
                         displays a dataframe that shows the duplicated rows between the datasets

        Raises:
            DeepchecksValueError: If the object is not a Dataset instance
        """
        return self._data_sample_leakage_report(validation_dataset=validation_dataset, train_dataset=train_dataset)

    def _data_sample_leakage_report(self, validation_dataset: Dataset, train_dataset: Dataset):
        validation_dataset = Dataset.validate_dataset_or_dataframe(validation_dataset)
        train_dataset = Dataset.validate_dataset_or_dataframe(train_dataset)
        validation_dataset.validate_shared_features(train_dataset, self.__class__.__name__)

        columns = train_dataset.features()
        if train_dataset.label_name():
            columns = columns + [train_dataset.label_name()]

        train_f = train_dataset.data.copy()
        val_f = validation_dataset.data.copy()

        train_dups = get_dup_indexes_map(train_f, columns)
        train_f.index = [f'Train indices: {get_dup_txt(i, train_dups)}' for i in train_f.index]
        train_f.drop_duplicates(columns, inplace=True)

        val_dups = get_dup_indexes_map(val_f, columns)
        val_f.index = [f'Validation indices: {get_dup_txt(i, val_dups)}' for i in val_f.index]
        val_f.drop_duplicates(columns, inplace=True)

        appended_df = train_f.append(val_f)
        duplicate_rows_df = appended_df[appended_df.duplicated(columns, keep=False)]
        duplicate_rows_df.sort_values(columns, inplace=True)

        count_val_array = np.zeros((duplicate_rows_df.shape[0],))
        idx_in_array = 0
        for index in duplicate_rows_df.index:
            if index.startswith('Validation'):
                if not 'Tot.' in index:
                    count_val_array[idx_in_array] = len(index.split(','))
                else:
                    count_val_array[idx_in_array] = int(re.findall(r'Tot. (\d+)', index)[0])
                count_val_array[idx_in_array + 1] = count_val_array[idx_in_array]
                idx_in_array += 2

        duplicate_rows_df = duplicate_rows_df.iloc[np.flip(count_val_array.argsort()), :]

        count_dups = count_val_array.sum() // 2

        dup_ratio = count_dups / validation_dataset.n_samples()
        user_msg = f'{format_percent(dup_ratio)} ({count_dups} / {validation_dataset.n_samples()}) \
                     of validation data samples appear in train data'
        display = [user_msg, duplicate_rows_df.head(10)] if dup_ratio else None

        return CheckResult(dup_ratio, header='Data Sample Leakage Report', check=self.__class__, display=display)
