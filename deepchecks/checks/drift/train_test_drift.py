"""Module contains Train Test Drift check."""

from collections import Counter
from typing import Union, Iterable, Tuple, List, Dict

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from deepchecks import Dataset, CheckResult, TrainTestBaseCheck, ConditionResult
import matplotlib.pyplot as plt

__all__ = ['TrainTestDrift']

PSI_MIN_PERCENTAGE = 0.01


def preprocess_for_psi(dist1: np.array, dist2: np.array, max_num_categories) -> Tuple[float, float, List]:
    """
    Preprocess distributions in order to be able to be calculated by PSI.

    Function returns the value counts for each distribution and the categories list. If there are more than
    max_num_categories, it encodes rare categories into an "OTHER" category. This is done according to the values of
    dist1, which is treated as the "expected" distribution.

    Function is for categorical data only.
    Args:
        dist1: first distribution, treated as the expected distribution
        dist2: second distribution, treated as the actual distribution
        max_num_categories: max number of allowed categories. If there are more, they are binned into an "Other"
        category. If max_num_categories=None, there is no limit.

    Returns:
        expected_percents: array of percentages of each value in the expected distribution.
        actual_percents: array of percentages of each value in the actual distribution.
        categories_list: list of all categories that the percentages represent.

    """
    all_categories = list(set(np.unique(dist1)).union(set(dist2)))

    if max_num_categories is not None and len(all_categories) > max_num_categories:
        dist1_counter = dict(Counter(dist1).most_common(max_num_categories))
        dist1_counter['Other rare categories'] = len(dist1) - sum(dist1_counter.values())
        categories_list = list(dist1_counter.keys())

        dist2_counter = Counter(dist2)
        dist2_counter = {k: dist2_counter[k] for k in categories_list}
        dist2_counter['Other rare categories'] = len(dist2) - sum(dist2_counter.values())

    else:
        dist1_counter = Counter(dist1)
        dist2_counter = Counter(dist2)
        categories_list = all_categories

    expected_percents = np.array([dist1_counter[k] for k in categories_list]) / len(dist1)
    actual_percents = np.array([dist2_counter[k] for k in categories_list]) / len(dist2)

    return expected_percents, actual_percents, categories_list


def psi(expected_percents: np.array, actual_percents: np.array):
    """
    Calculate the PSI (Population Stability Index).

    See https://www.lexjansen.com/wuss/2017/47_Final_Paper_PDF.pdf

    Args:
        expected_percents: array of percentages of each value in the expected distribution.
        actual_percents: array of percentages of each value in the actual distribution.

    Returns:
        psi: The PSI score

    """
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
    Calculate the Earth Movers Distance (Wasserstein distance).

    See https://en.wikipedia.org/wiki/Wasserstein_metric

    Function is for numerical data only.

    Args:
        dist1: array of numberical values.
        dist2: array of numberical values to compare dist1 to.

    Returns:
        the Wasserstein distance between the two distributions.

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
    """
    Calculate drift between train dataset and test dataset.

    Check calculates a drift score for each column in test dataset, by comparing its distribution to the train
    dataset.
    For numerical columns, we use the Earth Movers Distance.
    See https://www.lexjansen.com/wuss/2017/47_Final_Paper_PDF.pdf
    For categorical columns, we use the Population Stability Index (PSI).
    See https://en.wikipedia.org/wiki/Wasserstein_metric.
    """

    def __init__(self, columns: Union[str, Iterable[str]] = None, ignore_columns: Union[str, Iterable[str]] = None,
                 max_num_categories: int = 10):
        """
        Initialize the TrainTestDrift class.

        Args:
            columns (Union[str, Iterable[str]]): Columns to check, if none are given checks all columns except ignored
            ones.
            ignore_columns (Union[str, Iterable[str]]): Columns to ignore, if none given checks based on columns
            variable.
            max_num_categories (int): Only for categorical columns. Max number of allowed categories. If there are more,
             they are binned into an "Other" category. If max_num_categories=None, there is no limit.
        """
        super().__init__()
        self.columns = columns
        self.ignore_columns = ignore_columns
        self.max_num_categories = max_num_categories

    def run(self, train_dataset, test_dataset, model=None) -> CheckResult:
        """Run check.

        Args:
            train_dataset (Dataset): The training dataset object. Must contain a label column.
            test_dataset (Dataset): The test dataset object. Must contain a label column.
            model: A scikit-learn-compatible fitted estimator instance

        Returns:
            CheckResult:
                value: dictionary of column name to drift score.
                display: distribution graph for each column, comparing the train and test distributions.

        Raises:
            DeepchecksValueError: If the object is not a Dataset or DataFrame instance
        """
        return self._calc_drift(train_dataset, test_dataset)

    def _calc_drift(self, train_dataset: Dataset, test_dataset: Dataset) -> CheckResult:
        """
        Calculate drift for all columns.

        Args:
            train_dataset (Dataset): The training dataset object. Must contain a label column.
            test_dataset (Dataset): The test dataset object. Must contain a label column.

        Returns:
            CheckResult:
                value: dictionary of column name to drift score.
                display: distribution graph for each column, comparing the train and test distributions.
        """
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
            value={column: {'value': result[0], 'method': result[1]} for column, result in zip(features, results)},
            # display=pd.DataFrame(results, index=['Drift result']).T,
            display=[result[2] for result in results],
            header='Train Test Drift',
            check=self.__class__
        )

    def _calc_drift_per_column(self, train_column: pd.Series, test_column: pd.Series, column_name: str,
                               column_type: str):
        """
        Calculate drift score per column.

        Args:
            train_column: column from train dataset
            test_column: same column from test dataset
            column_name: name of column
            column_type: type of column (either "numerical" or "categorical")

        Returns:
            score: drift score of the difference between the two columns' distributions (Earth movers distance for
            numerical, PSI for categorical)
            display: graph comparing the two distributions (density for numerical, stack bar for categorical)
        """
        train_dist = train_column.dropna().values.reshape(-1)
        test_dist = test_column.dropna().values.reshape(-1)

        if column_type == 'numerical':
            score = earth_movers_distance(dist1=train_column.astype('float'), dist2=test_column.astype('float'))

            def plot_numerical():
                plt.title(f'Distribution of {column_name}')
                train_column.plot(kind='density', label='Train dataset', legend=True, figsize=(8, 4))
                test_column.plot(kind='density', label='Test dataset', legend=True, figsize=(8, 4))

            return score, "Earth Mover's Distance", plot_numerical

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

            return score, 'PSI', plot_categorical

    def add_condition_drift_score_not_greater_than(self, max_allowed_psi_score: float = 0.2,
                                                   max_allowed_earth_movers_score: float = 0.1):
        """
        Add condition - require drift score to not be more than a certain threshold.

        The industry standard for PSI limit is above 0.2.
        Earth movers does not have a common industry standard.

        Args:
            max_allowed_psi_score: the max threshold for the PSI score
            max_allowed_earth_movers_score: the max threshold for the Earth Mover's Distance score

        Returns:
            ConditionResult: False if any column has passed the max threshold, True otherwise
        """

        def condition(result: Dict) -> ConditionResult:
            not_passing_categorical_columns = [column for column, d in result.items() if
                                               d['value'] > max_allowed_psi_score and d['method'] == 'PSI']
            not_passing_numeric_columns = [column for column, d in result.items() if
                                           d['value'] > max_allowed_earth_movers_score
                                           and d['method'] == "Earth Mover's Distance"]
            return_str = ''
            if not_passing_categorical_columns:
                return_str += f'Found categorical columns with PSI over {max_allowed_psi_score}: ' \
                              f'{", ".join(not_passing_categorical_columns)}\n'
            if not_passing_numeric_columns:
                return_str += f'Found numeric columns with Earth Mover\'s Distance over ' \
                              f'{max_allowed_earth_movers_score}: {", ".join(not_passing_numeric_columns)}'

            if return_str:
                return ConditionResult(False, return_str)
            else:
                return ConditionResult(True)

        return self.add_condition(f'PSI and Earth Mover\'s Distance cannot be greater than {max_allowed_psi_score} and '
                                  f'{max_allowed_earth_movers_score} respectively',
                                  condition)
