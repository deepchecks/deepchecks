from collections import Counter
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import LabelEncoder

from mlchecks import Dataset, CompareDatasetsBaseCheck, CheckResult
from mlchecks.utils import MLChecksValueError, get_plt_base64, get_plt_html_str
from logging import getLogger

logger = getLogger("dataset_drift")


def _prepare_series(dataset: pd.DataFrame, compared_dataset: pd.DataFrame, column_name: str):
    dataset_col = dataset[column_name].dropna().values.reshape(-1)
    comp_dataset_col = compared_dataset[column_name].dropna().values.reshape(-1)

    return dataset_col, comp_dataset_col


def _psi(dataset: Dataset, compared_dataset: Dataset, column_name: str) -> float:

    dataset_col, compared_dataset_col = _prepare_series(dataset, compared_dataset, column_name)

    sample_space = list(set(dataset_col).union(set(compared_dataset_col)))
    encoder = LabelEncoder().fit(sample_space)
    dataset_col = encoder.transform(dataset_col)
    compared_dataset_col = encoder.transform(compared_dataset_col)

    expected_array = compared_dataset_col.astype('float')
    actual_array = dataset_col.astype('float')

    if (len(expected_array) < 1) or (len(actual_array) < 1):
        return 0
    sample_space = list(set(expected_array).union(set(actual_array)))

    expected_dict = dict(Counter(expected_array))
    expected_percents = np.array([expected_dict[k] if k in expected_dict else 0 for k in sample_space]) \
                        / len(expected_array)

    actual_dict = dict(Counter(actual_array))
    actual_percents = np.array([actual_dict[k] if k in actual_dict else 0 for k in sample_space]) \
                      / len(actual_array)

    psi_value = 0
    for i in range(0, len(expected_percents)):
        MIN_PERC = 0.1

        a_perc = max(actual_percents[i], MIN_PERC)
        e_perc = max(expected_percents[i], MIN_PERC)

        psi_value += (e_perc - a_perc) * np.log(e_perc / a_perc)

    return psi_value


def _wass_distance(dataset: pd.DataFrame, compared_dataset: pd.DataFrame,column_name: str) -> float:
    """ compute earth movers distance using dit earth_movers_distance"""
    dataset_col, compared_dataset_col = _prepare_series(dataset, compared_dataset, column_name)

    # count each unique value
    unique1, counts1 = np.unique(dataset_col, return_counts=True)
    unique2, counts2 = np.unique(compared_dataset_col, return_counts=True)

    sample_space = list(set(unique1).union(set(unique2)))

    # if numeric, minmax scale to ensure that the result would be bound [0,1]
    val_max = max(sample_space)
    val_min = min(sample_space)
    dataset_col = (dataset_col - val_min) / val_max
    compared_dataset_col = (compared_dataset_col - val_min) / val_max

    return wasserstein_distance(dataset_col, compared_dataset_col)

def _overtime_numerical_dist(dataset: pd.DataFrame, compared_dataset: pd.DataFrame, column_name: str):
    pass

def _overtime_categorical_dist(dataset: pd.DataFrame, compared_dataset: pd.DataFrame, column_name: str):
    pass

def _static_numerical_dist(dataset: pd.DataFrame, compared_dataset: pd.DataFrame, column_name: str):
    dataset[column_name].plot(kind='density', label='Dataset')

    try:
        compared_dataset[column_name].plot(kind='density', label="Compared Dataset")
    except:
        logger.info(f"Unable to draw compared_dataset plot for {column_name}")

    return get_plt_html_str()

def _static_categorical_dist(dataset: pd.DataFrame, compared_dataset: pd.DataFrame, column_name: str):
    pass

def dataset_drift(dataset: Dataset,
                  compared_dataset: Dataset,
                  column_name: str = None,
                  over_time: bool = False) -> CheckResult:

    if not isinstance(dataset, Dataset):
        raise MLChecksValueError("dataset must be of instance Dataset")

    if not isinstance(compared_dataset, Dataset):
        raise MLChecksValueError("compared_dataset must be of instance Dataset")

    if column_name:
        if column_name not in set(dataset.columns) or column_name not in set(compared_dataset.columns):
            raise MLChecksValueError(f'Column name {column_name} must exist in both datasets')

    if over_time:
        if not dataset.date_col():
            raise MLChecksValueError("dataset does not contain a date column and over_time=True")

        if not compared_dataset.date_col():
            raise MLChecksValueError("compared_dataset does not contain a date column and over_time=True")

    all_columns = column_name if column_name is not None else set(dataset.columns)
    categorical_features = set(dataset.cat_features())

    comp_all_columns = set(compared_dataset.columns)
    comp_cat_features = set(compared_dataset.features())
    display_items = []
    for col in all_columns:
        calculate_drift = False
        if col not in comp_all_columns:
            logger.warning(f"The column {col} does not exist in the compared_dataset. "
                           f"drift calculation for this column will be skipped")
        elif col in categorical_features and col not in comp_cat_features:
            logger.warning(f"The column {col} is categorical in the dataset but not categorical in the "
                           f"compared_dataset. drift calculation for this column will be skipped")
        elif col not in categorical_features and col in comp_cat_features:
            logger.warning(f"The column {col} is not categorical in the dataset but categorical in the "
                           f"compared_dataset. drift calculation for this column will be skipped")
        else:
            calculate_drift = True

        if over_time:
            if col not in categorical_features:
                image = _overtime_numerical_dist(dataset, compared_dataset, col)
            else:
                image = _overtime_categorical_dist(dataset, compared_dataset, col)
        else:
            if col not in categorical_features:
                image = _static_numerical_dist(dataset, compared_dataset, col)
            else:
                image = _static_categorical_dist(dataset, compared_dataset, col)
        # Generate the distribution chart
        display_items.append(image)
        # If we can calculate drift
        if calculate_drift:
            # TODO: calculate the drift based on over time
            pass







class DatasetDrift(CompareDatasetsBaseCheck):
    def run(self, dataset, compared_dataset, model=None) -> CheckResult:
        column_name = self.params.get('column_name')
        over_time = self.params.get('over_time')

        return dataset_drift(dataset, compared_dataset, column_name, over_time)
