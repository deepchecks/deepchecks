"""The data_sample_leakage_report check module."""
from typing import Dict, List
import re

from mlchecks import Dataset
from mlchecks.base.check import CheckResult, TrainValidationBaseCheck
from mlchecks.string_utils import format_percent


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
    """Return a prettyfied text for a key in the dict.

    Args:
        i: the index key
        dup_map: the dict of the duplicated indexes
    Returns:
        prettyfied text for a key in the dict

    """
    val = dup_map.get(i)
    if not val:
        return i
    txt = f'{i}, '
    for j in val:
        txt += f'{j}, '
    txt = txt[:-2]
    if len(txt) < 30:
        return txt
    return f'{txt[:30]}.. Tot. {(1 + len(val))}'


class DataSampleLeakageReport(TrainValidationBaseCheck):
    """Find what percent of the validation data is in the train data"""

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
            MLChecksValueError: If the object is not a Dataset instance
        """
        return self._data_sample_leakage_report(validation_dataset=validation_dataset, train_dataset=train_dataset)

    def _data_sample_leakage_report(self, validation_dataset: Dataset, train_dataset: Dataset):
        validation_dataset = Dataset.validate_dataset_or_dataframe(validation_dataset)
        train_dataset = Dataset.validate_dataset_or_dataframe(train_dataset)
        validation_dataset.validate_shared_features(train_dataset, self.__class__.__name__)

        columns = train_dataset.features()
        if train_dataset.label_name():
            columns = columns + [train_dataset.label_name()]

        train_f = train_dataset.data
        val_f = validation_dataset.data

        train_dups = get_dup_indexes_map(train_f, columns)
        train_f.index = [f'Train indexes: {get_dup_txt(i, train_dups)}' for i in train_f.index]
        train_f.drop_duplicates(columns, inplace=True)

        val_dups = get_dup_indexes_map(val_f, columns)
        val_f.index = [f'Validation indexes: {get_dup_txt(i, val_dups)}' for i in val_f.index]
        val_f.drop_duplicates(columns, inplace=True)

        appended_df = train_f.append(val_f)
        duplicate_rows_df = appended_df[appended_df.duplicated(columns, keep=False)]
        duplicate_rows_df.sort_values(columns, inplace=True)

        count_dups = 0
        for index in duplicate_rows_df.index:
            if index.startswith('Train'):
                if not 'Tot.' in index:
                    count_dups += len(index.split(','))
                else:
                    count_dups += int(re.findall(r'Tot. (\d+)', index)[0])

        dup_ratio = count_dups / len(val_f)
        user_msg = f'{format_percent(dup_ratio)} ({count_dups} / {len(val_f)}) \
                     of validation data samples appear in train data'
        display = [user_msg, duplicate_rows_df.head(10)] if dup_ratio else None

        return CheckResult(dup_ratio, header='Data Sample Leakage Report', check=self.__class__, display=display)
