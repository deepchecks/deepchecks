"""The data_sample_leakage_report check module."""
from typing import Dict, List

import pandas as pd
from mlchecks import Dataset

from mlchecks.base.check import CheckResult, TrainValidationBaseCheck

__all__ = ['data_sample_leakage_report', 'DataSampleLeakageReport']


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
    return txt[:-2]


def data_sample_leakage_report(validation_dataset: Dataset, train_dataset: Dataset):
    """Find which percent of the validation data in the train data.

    Args:
        train_dataset (Dataset): The training dataset object. Must contain an index.
        validation_dataset (Dataset): The validation dataset object. Must contain an index.
    Returns:
        CheckResult: value is sample leakage ratio in %,
                     displays a dataframe that shows the duplicated rows between the datasets

    Raises:
        MLChecksValueError: If the object is not a Dataset instance

    """
    validation_dataset = Dataset.validate_dataset_or_dataframe(validation_dataset)
    train_dataset = Dataset.validate_dataset_or_dataframe(train_dataset)
    validation_dataset.validate_shared_features(train_dataset, data_sample_leakage_report.__name__)

    columns = train_dataset.features()
    if train_dataset.label_name():
        columns = columns + [train_dataset.label_name()]
    
    train_f = train_dataset.data
    val_f = validation_dataset.data

    train_dups = get_dup_indexes_map(train_f, columns)
    train_f.index = [f'test indexes: {get_dup_txt(i, train_dups)}' for i in train_f.index]
    train_f.drop_duplicates(columns, inplace = True)

    val_dups = get_dup_indexes_map(val_f, columns)
    val_f.index = [f'validation indexes: {get_dup_txt(i, val_dups)}' for i in val_f.index]
    val_f.drop_duplicates(columns, inplace = True)

    appended_df = train_f.append(val_f)
    duplicate_rows_df = appended_df[appended_df.duplicated(columns, keep=False)]
    duplicate_rows_df.sort_values(columns, inplace=True)

    count_dups = 0
    for index in duplicate_rows_df.index:
        if not index.startswith('test'):
            continue
        count_dups += len(index.split(','))

    dup_ratio = count_dups / len(val_f) * 100
    user_msg = f'You have {dup_ratio:0.2f}% ({count_dups} / {len(val_f)}) \
                 of the validation data samples appear in train data'
    display = [user_msg, duplicate_rows_df.head(10)] if dup_ratio else None

    return CheckResult(dup_ratio, header='Data Sample Leakage Report',
                       check=data_sample_leakage_report, display=display)

class DataSampleLeakageReport(TrainValidationBaseCheck):
    """Finds data sample leakage."""

    def run(self, validation_dataset: Dataset, train_dataset: Dataset) -> CheckResult:
        """Run data_sample_leakage_report check.

        Args:
            train_dataset (Dataset): The training dataset object. Must contain an index.
            validation_dataset (Dataset): The validation dataset object. Must contain an index.
        Returns:
            CheckResult: value is sample leakage ratio in %,
                         displays a dataframe that shows the duplicated rows between the datasets
        """
        return data_sample_leakage_report(validation_dataset=validation_dataset, train_dataset=train_dataset)
