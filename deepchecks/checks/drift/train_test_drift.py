"""Module contains Train Test Drift check."""

from collections import Counter
from typing import Union, Iterable

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from deepchecks import Dataset, CheckResult, TrainTestBaseCheck
import matplotlib.pyplot as plt

__all__ = ['TrainTestDrift']

PSI_MIN_PERCENTAGE = 0.01


def preprocess_for_psi(dist1: np.array, dist2: np.array, max_num_categories):
    """
    Preprocess distributions in order to be able to be sent f

    Function preprocesses the data so it encodes rare categories into an "OTHER" category. This is done according to
    the values of dist1, which is treated as the "expected" distribution.

    Function is for categorical data only.
    """
    all_categories = list(set(np.unique(dist1)).union(set(dist2)))

    if max_num_categories is not None and len(all_categories) > max_num_categories:
        dist1_counter = dict(Counter(dist1).most_common(max_num_categories))
        dist1_counter['OTHER_RARE_CATEGORIES'] = len(dist1) - sum(dist1_counter.values())
        categories_list = list(dist1_counter.keys())

        dist2_counter = Counter(dist2)
        dist2_counter = {k: dist2_counter[k] for k in categories_list}
        dist2_counter['OTHER_RARE_CATEGORIES'] = len(dist2) - sum(dist2_counter.values())

    else:
        dist1_counter = Counter(dist1)
        dist2_counter = Counter(dist2)
        categories_list = all_categories

    expected_percents = np.array([dist1_counter[k] for k in categories_list]) / len(dist1)
    actual_percents = np.array([dist2_counter[k] for k in categories_list]) / len(dist2)

    return expected_percents, actual_percents, categories_list


def psi(expected_percents: np.array, actual_percents: np.array):
    psi_value = 0
    for i in range(len(expected_percents)):
        # In order for the value not to diverge, we cap our min percentage value
        e_perc = max(expected_percents[i], PSI_MIN_PERCENTAGE)
        a_perc = max(actual_percents[i], PSI_MIN_PERCENTAGE)
        value = (e_perc - a_perc) * np.log(e_perc / a_perc)
        psi_value += value

    return psi_value


def earth_movers_distance(dist1: np.array, dist2: np.array):
    """
    Function is for numerical data only
    """
    unique1 = np.unique(dist1)
    unique2 = np.unique(dist2)

    sample_space = list(set(unique1).union(set(unique2)))

    val_max = max(sample_space)
    val_min = min(sample_space)

    if val_max == val_min:
        return 0

    dist1 = (dist1 - val_min) / (val_max - val_min)
    dist2 = (dist2 - val_min) / (val_max - val_min)

    return wasserstein_distance(dist1, dist2)


class TrainTestDrift(TrainTestBaseCheck):
    """Calculate drift between train dataset and test dataset"""

    def __init__(self, columns: Union[str, Iterable[str]] = None, ignore_columns: Union[str, Iterable[str]] = None,
                 max_num_categories: int = 10):
        """
        Initialize the TrainTestDrift class.

        Args:
            columns (Union[str, Iterable[str]]): Columns to check, if none are given checks all columns except ignored
            ones.
            ignore_columns (Union[str, Iterable[str]]): Columns to ignore, if none given checks based on columns
            variable.
        """
        super().__init__()
        self.columns = columns
        self.ignore_columns = ignore_columns
        self.max_num_categories = max_num_categories

    def run(self, train_dataset, test_dataset, model=None) -> CheckResult:

        return self._calc_drift(train_dataset, test_dataset)

    def _calc_drift(self, train_dataset: Dataset, test_dataset: Dataset) -> CheckResult:

        train_dataset = Dataset.validate_dataset_or_dataframe(train_dataset)
        test_dataset = Dataset.validate_dataset_or_dataframe(test_dataset)

        train_dataset = train_dataset.filter_columns_with_validation(self.columns, self.ignore_columns)
        test_dataset = test_dataset.filter_columns_with_validation(self.columns, self.ignore_columns)

        features = train_dataset.validate_shared_features(test_dataset, self.__class__.__name__)
        cat_features = train_dataset.validate_shared_categorical_features(test_dataset, self.__class__.__name__)

        results = [self._calc_drift_per_column(train_column=train_dataset.data[column],
                                               test_column=test_dataset.data[column],
                                               column_name=column,
                                               column_type='categorical' if column in cat_features else 'numerical')
                   for column in features
                   ]

        return CheckResult(
            value={column: result[0] for column, result in zip(features, results)},
            # display=pd.DataFrame(results, index=['Drift result']).T,
            display=[result[1] for result in results],
            header='Train Test Drift',
            check=self.__class__
        )

    def _calc_drift_per_column(self, train_column: pd.Series, test_column: pd.Series, column_name: str,
                               column_type: str):
        train_dist = train_column.dropna().values.reshape(-1)
        test_dist = test_column.dropna().values.reshape(-1)

        if column_type == 'numerical':
            score = earth_movers_distance(dist1=train_column.astype('float'), dist2=test_column.astype('float'))

            def plot_numerical():
                plt.title(f'Distribution of {column_name}')
                train_column.plot(kind='density', label='Train dataset', legend=True, figsize=(8,4))
                test_column.plot(kind='density', label='Test dataset', legend=True, figsize=(8,4))

            return score, plot_numerical

        elif column_type == 'categorical':

            expected_percents, actual_percents, categories_list = \
                preprocess_for_psi(dist1=train_dist, dist2=test_dist, max_num_categories=self.max_num_categories)
            score = psi(expected_percents=expected_percents, actual_percents=actual_percents)

            def plot_categorical():

                labels = ['Train dataset', 'Test dataset']
                width = 0.35

                fig, ax = plt.subplots()
                fig.set_size_inches(8, 4)

                expected_bar_height = 0
                actual_bar_height = 0
                for expected, actual, category in zip(expected_percents, actual_percents, categories_list):
                    ax.bar(labels, [expected, actual], width, label=category,
                           bottom=[expected_bar_height, actual_bar_height])
                    expected_bar_height += expected
                    actual_bar_height += actual

                ax.set_ylabel('Percentage')
                ax.set_title(f'Distribution of {column_name}')
                ax.legend()

            return score, plot_categorical


