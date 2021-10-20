"""The data_sample_leakage_report check module."""
from typing import Dict
from mlchecks import Dataset

from mlchecks.base.check import CheckResult, TrainValidationBaseCheck
from mlchecks.base.dataset import validate_dataset

__all__ = ['data_sample_leakage_report', 'DataSampleLeakageReport']


def get_dup_indexes_map(df, features) -> Dict:
    """Find duplicated indexes in the dataframe.

    Args:
        df: a Dataframe object
        features: the features name list
    Returns:
        dictionary of each of the first indexes and its' duplicated indexes

    """
    dup = df[df.duplicated(features, keep=False)].groupby(features).groups.values()
    dup_map = {}
    for i_arr in dup:
        key = i_arr[0]
        dup_map[key] = [int(i) for i in i_arr[1:]]
    return dup_map


def get_dup_txt(i, dup_map) -> str:
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
    validate_dataset(validation_dataset, data_sample_leakage_report.__name__)
    validate_dataset(train_dataset, data_sample_leakage_report.__name__)

    features = train_dataset.features()
    train_f = train_dataset[features]
    val_f = validation_dataset[features]

    train_dups = get_dup_indexes_map(train_dataset, features)
    train_f.index = [f'test indexs: {get_dup_txt(i, train_dups)}' for i in train_f.index]
    train_f.drop_duplicates(features, inplace = True)

    val_dups = get_dup_indexes_map(val_f, features)
    val_f.index = [f'validation indexs: {get_dup_txt(i, val_dups)}' for i in val_f.index]
    val_f.drop_duplicates(features, inplace = True)

    appended_df = train_f.append(val_f)
    duplicate_rows_df = appended_df[appended_df.duplicated(features, keep=False)]
    duplicate_rows_df.sort_values(features, inplace=True)

    count_dups = 0
    for index in duplicate_rows_df.index:
        if not index.startswith('test'):
            continue
        count_dups += len(index.split(','))

    dup_ratio = count_dups / len(val_f) * 100
    user_msg = 'You have {0:0.2f}% of the validation data in the train data.'.format(dup_ratio) # pylint: disable=locally-disabled, consider-using-f-string

    display = [user_msg, duplicate_rows_df] if dup_ratio else None

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
