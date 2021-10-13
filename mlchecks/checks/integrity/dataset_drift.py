from collections import Counter
from typing import Union, List

import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib.dates import DateFormatter
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import LabelEncoder

from mlchecks import Dataset, CompareDatasetsBaseCheck, CheckResult
from mlchecks.utils import MLChecksValueError, get_plt_html_str
from logging import getLogger
import matplotlib.pyplot as plt

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

    if len(dataset) == 0 or len(compared_dataset) == 0:
        return 0
    return wasserstein_distance(dataset_col, compared_dataset_col)


def _draw_overtime_results(comp_dist: pd.DataFrame,
                           overtime_dist: pd.DataFrame,
                           column_name: str,
                           drift_res: pd.DataFrame = None):
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 3])
    ax1 = plt.subplot(gs[0])

    compared_dataset_stats = (comp_dist[column_name].value_counts() / comp_dist.shape[0]).reset_index()

    compared_dataset_stats.columns = ['category', 'percent']
    compared_dataset_stats['Compared Dataset'] = ""
    compared_dataset_stats = compared_dataset_stats.set_index(['Compared Dataset', 'category']).unstack('category')

    compared_dataset_stats.plot(kind='bar', stacked=True, ax=ax1)

    ax2 = plt.subplot(gs[1])
    overtime_dist.plot(kind='bar', stacked=True, ax=ax2)
    if drift_res is not None:
        twin1 = ax2.twinx()
        p2, = twin1.plot(drift_res, "r-", label="Drift Score")
        twin1.yaxis.label.set_color(p2.get_color())

    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.title(f"Distribution plot of feature {column_name} over time")
    fig.autofmt_xdate()
    ax2.fmt_xdata = DateFormatter('%Y-%m-%d')

    return get_plt_html_str()


def overtime_numerical_dist(dataset: Dataset, compared_dataset: Dataset, column_name: str, calculate_drift=False):
    percentiles = compared_dataset[column_name].quantile(np.linspace(.1, 1, 9, 0))
    compared_dataset[f'{column_name}_percentiles'] = pd.qcut(compared_dataset[column_name], 10, labels=False)
    dataset[f'{column_name}_percentiles'] = pd.cut(dataset[column_name], percentiles, labels=False, include_lowest=True)

    orig_col_name = column_name
    column_name = f'{column_name}_percentiles'
    grouped_by = dataset.groupby(pd.Grouper(freq='W', key='date'))
    stats_df = grouped_by[column_name].apply(
        lambda ser: ser.value_counts() / ser.shape[0]).reset_index()

    stats_df.columns = ['date', 'category', 'percent']
    stats_df = stats_df.set_index(['date', 'category']).unstack('category')
    drifts_df = None

    if calculate_drift:
        drifts_df = grouped_by.apply(lambda x: _wass_distance(x, compared_dataset, orig_col_name))
        drifts_df = drifts_df[stats_df.index]

    html_plot = _draw_overtime_results(compared_dataset, stats_df, column_name, drifts_df)

    return html_plot


def overtime_categorical_dist(dataset: Dataset, compared_dataset: Dataset, column_name: str, calculate_drift=False):
    grouped_by = dataset.groupby(pd.Grouper(freq='W', key='date'))
    stats_df = grouped_by[column_name].apply(
        lambda ser: ser.value_counts() / ser.shape[0]).reset_index()

    stats_df.columns = ['date', 'category', 'percent']
    stats_df = stats_df.set_index(['date', 'category']).unstack('category')
    drifts_df = None

    if calculate_drift:
        drifts_df = grouped_by.apply(lambda x: _psi(x, compared_dataset, column_name))
        drifts_df = drifts_df[stats_df.index]

    html_plot = _draw_overtime_results(compared_dataset, stats_df, column_name, drifts_df)

    return html_plot


def _static_numerical_dist(dataset: pd.DataFrame, compared_dataset: pd.DataFrame, column_name: str):
    dataset[column_name].plot(kind='density', label='Dataset')

    try:
        compared_dataset[column_name].plot(kind='density', label="Compared Dataset")
    except:
        logger.info(f"Unable to draw compared_dataset distribution plot for {column_name}")

    return get_plt_html_str()


def _static_categorical_dist(dataset: pd.DataFrame, compared_dataset: pd.DataFrame, column_name: str):
    pass


def dataset_drift(dataset: Dataset,
                  compared_dataset: Dataset,
                  column_names: Union[List[str], str] = None,
                  over_time: bool = False) -> CheckResult:

    if not isinstance(dataset, Dataset):
        raise MLChecksValueError("dataset must be of instance Dataset")

    if not isinstance(compared_dataset, Dataset):
        raise MLChecksValueError("compared_dataset must be of instance Dataset")

    if column_names:
        if column_names not in set(dataset.columns) or column_names not in set(compared_dataset.columns):
            raise MLChecksValueError(f'Column name {column_names} must exist in both datasets')

    if over_time:
        if not dataset.date_col():
            raise MLChecksValueError("dataset does not contain a date column and over_time=True")

        if not compared_dataset.date_col():
            raise MLChecksValueError("compared_dataset does not contain a date column and over_time=True")

    all_columns = column_names if column_names is not None else set(dataset.columns)
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
                image = overtime_categorical_dist(dataset, compared_dataset, col)
        else:
            if col not in categorical_features:
                image = _static_numerical_dist(dataset, compared_dataset, col)
            else:
                image = _static_categorical_dist(dataset, compared_dataset, col)
        # Generate the distribution chart
        display_items.append(image)
        # If we can calculate drift


class DatasetDrift(CompareDatasetsBaseCheck):
    def run(self, dataset, compared_dataset, model=None) -> CheckResult:
        column_names = self.params.get('column_names')
        over_time = self.params.get('over_time')

        return dataset_drift(dataset, compared_dataset, column_names, over_time)
