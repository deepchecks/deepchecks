# ----------------------------------------------------------------------------
# Copyright (C) 2021 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""Module contains Train Test Drift check."""

from collections import Counter, OrderedDict
from typing import Union, Tuple, List, Dict, Callable

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from deepchecks import Dataset, CheckResult, TrainTestBaseCheck, ConditionResult
from deepchecks.checks.distribution.plot import plot_density
from deepchecks.utils.features import calculate_feature_importance_or_null
from deepchecks.utils.typing import Hashable
from deepchecks.errors import DeepchecksValueError
import matplotlib.pyplot as plt

__all__ = ['TrainTestDrift']


PSI_MIN_PERCENTAGE = 0.01


def preprocess_for_psi(dist1: np.ndarray, dist2: np.ndarray, max_num_categories) -> Tuple[np.ndarray, np.ndarray, List]:
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


def psi(expected_percents: np.ndarray, actual_percents: np.ndarray):
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


def earth_movers_distance(dist1: np.ndarray, dist2: np.ndarray):
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


    Args:
        columns (Union[Hashable, List[Hashable]]):
            Columns to check, if none are given checks all
            columns except ignored ones.
        ignore_columns (Union[Hashable, List[Hashable]]):
            Columns to ignore, if none given checks based on
            columns variable.
        n_top_columns (int): (optional - used only if model was specified)
            amount of columns to show ordered by feature importance (date, index, label are first)
        sort_feature_by (str):
            Indicates how features will be sorted. Can be either "feature importance"
            or "drift score"
        max_num_categories (int):
            Only for categorical columns. Max number of allowed categories. If there are more,
            they are binned into an "Other" category. If max_num_categories=None, there is no limit.
    """

    def __init__(
        self,
        columns: Union[Hashable, List[Hashable], None] = None,
        ignore_columns: Union[Hashable, List[Hashable], None] = None,
        n_top_columns: int = 5,
        sort_feature_by: str = 'feature importance',
        max_num_categories: int = 10
    ):
        super().__init__()
        self.columns = columns
        self.ignore_columns = ignore_columns
        self.max_num_categories = max_num_categories
        if sort_feature_by in {'feature importance', 'drift score'}:
            self.sort_feature_by = sort_feature_by
        else:
            raise DeepchecksValueError('sort_feature_by must be either "feature importance" or "drift score"')
        self.n_top_columns = n_top_columns

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
        feature_importances = calculate_feature_importance_or_null(train_dataset, model)
        return self._calc_drift(train_dataset, test_dataset, feature_importances)

    def _calc_drift(self, train_dataset: Dataset, test_dataset: Dataset, feature_importances: pd.Series) -> CheckResult:
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

        features = train_dataset.validate_shared_features(test_dataset)
        cat_features = train_dataset.validate_shared_categorical_features(test_dataset)

        values_dict = OrderedDict()
        displays_dict = OrderedDict()
        for column in features:
            value, method, display = self._calc_drift_per_column(
                train_column=train_dataset.data[column],
                test_column=test_dataset.data[column],
                column_name=column,
                column_type='categorical' if column in cat_features else 'numerical',
                feature_importances=feature_importances
            )
            values_dict[column] = {
                'Drift score': value,
                'Method': method,
                'Importance': feature_importances[column] if feature_importances is not None else None
            }
            displays_dict[column] = display

        if self.sort_feature_by == 'feature importance' and feature_importances is not None:
            columns_order = feature_importances.sort_values(ascending=False).head(self.n_top_columns).index
        else:
            columns_order = sorted(features, key=lambda col: values_dict[col]['Drift score'], reverse=True
                                   )[:self.n_top_columns]

        sorted_by = self.sort_feature_by if feature_importances is not None else 'drift score'

        headnote = f"""<span>
            The Drift score is a measure for the difference between two distributions, in this check - the test
            and train distributions.<br> The check shows the drift score and distributions for the features, sorted by
            {sorted_by} and showing only the top {self.n_top_columns} features, according to {sorted_by}.
            <br>If available, the plot titles also show the feature importance (FI) rank.
        </span>"""

        displays = [headnote] + [displays_dict[col] for col in columns_order]

        return CheckResult(value=values_dict, display=displays, header='Train Test Drift')

    def _calc_drift_per_column(self, train_column: pd.Series, test_column: pd.Series, column_name: Hashable,
                               column_type: str, feature_importances: pd.Series = None
                               ) -> Tuple[float, str, Callable]:
        """
        Calculate drift score per column.

        Args:
            train_column: column from train dataset
            test_column: same column from test dataset
            column_name: name of column
            column_type: type of column (either "numerical" or "categorical")
            feature_importances: feature importances series

        Returns:
            score: drift score of the difference between the two columns' distributions (Earth movers distance for
            numerical, PSI for categorical)
            display: graph comparing the two distributions (density for numerical, stack bar for categorical)
        """
        train_dist = train_column.dropna().values.reshape(-1)
        test_dist = test_column.dropna().values.reshape(-1)

        def drift_score_bar(axes, drift_score: float, drift_type: str):
            """Create a traffic light bar plot representing the drift score.

            Args:
                axes (): Matplotlib axes object
                drift_score (float): Drift score
                drift_type (str): The name of the drift metric used
            """
            stop = max(0.4, drift_score + 0.1)
            traffic_light_colors = [((0, 0.1), '#01B8AA'),
                                    ((0.1, 0.2), '#F2C80F'),
                                    ((0.2, 0.3), '#FE9666'),
                                    ((0.3, 1), '#FD625E')
                                    ]

            for range_tuple, color in traffic_light_colors:
                if range_tuple[0] <= drift_score < range_tuple[1]:
                    axes.barh(0, drift_score - range_tuple[0], left=range_tuple[0], color=color)
                elif drift_score >= range_tuple[1]:
                    axes.barh(0, range_tuple[1] - range_tuple[0], left=range_tuple[0], color=color)
            axes.set_title('Drift Score - ' + drift_type)
            axes.set_xlim([0, stop])
            axes.set_yticklabels([])

        colors = ['darkblue', '#69b3a2']

        if feature_importances is not None:
            fi_rank_series = feature_importances.rank(method='first', ascending=False)
            fi_rank = fi_rank_series[column_name]
            plot_title = f'{column_name} (#{int(fi_rank)} in FI)'
        else:
            plot_title = column_name

        if column_type == 'numerical':
            score = earth_movers_distance(dist1=train_column.astype('float'), dist2=test_column.astype('float'))

            def plot_numerical():

                x_range = (min(train_column.min(), test_column.min()), max(train_column.max(), test_column.max()))
                xs = np.linspace(x_range[0], x_range[1], 40)
                fig, axs = plt.subplots(3, figsize=(8, 4.5), gridspec_kw={'height_ratios': [1, 7, 0.2]})
                fig.suptitle(plot_title, horizontalalignment='left', fontweight='bold', x=0.05)
                drift_score_bar(axs[0], score, 'Earth Movers Distance')
                plt.sca(axs[1])
                pdf1 = plot_density(train_column, xs, colors[0])
                pdf2 = plot_density(test_column, xs, colors[1])
                plt.gca().set_ylim(bottom=0, top=max(max(pdf1), max(pdf2)) * 1.1)
                axs[1].set_xlabel(column_name)
                axs[1].set_ylabel('Probability Density')
                axs[1].legend(['Train dataset', 'Test Dataset'])
                axs[1].set_title('Distribution')
                fig.tight_layout(pad=1.0)
                axs[2].axhline(y=0.5, color='k', linestyle='-', linewidth=0.5)
                axs[2].axis('off')

            return score, "Earth Mover's Distance", plot_numerical

        elif column_type == 'categorical':

            expected_percents, actual_percents, categories_list = \
                preprocess_for_psi(dist1=train_dist, dist2=test_dist, max_num_categories=self.max_num_categories)
            score = psi(expected_percents=expected_percents, actual_percents=actual_percents)

            def plot_categorical():

                cat_df = pd.DataFrame({'Train dataset': expected_percents, 'Test dataset': actual_percents},
                                      index=categories_list)

                fig, axs = plt.subplots(3, figsize=(8, 4.5), gridspec_kw={'height_ratios': [1, 7, 0.2]})
                fig.suptitle(plot_title, horizontalalignment='left', fontweight='bold', x=0.05)
                drift_score_bar(axs[0], score, 'PSI')
                cat_df.plot.bar(ax=axs[1], color=colors)
                axs[1].set_ylabel('Percentage')
                axs[1].legend()
                axs[1].set_title('Distribution')
                plt.sca(axs[1])
                plt.xticks(rotation=30)
                fig.tight_layout(pad=1.0)
                axs[2].axhline(y=0.5, color='k', linestyle='-', linewidth=0.5)
                axs[2].axis('off')

            return score, 'PSI', plot_categorical

    def add_condition_drift_score_not_greater_than(self, max_allowed_psi_score: float = 0.2,
                                                   max_allowed_earth_movers_score: float = 0.1,
                                                   number_of_top_features_to_consider: int = 5):
        """
        Add condition - require drift score to not be more than a certain threshold.

        The industry standard for PSI limit is above 0.2.
        Earth movers does not have a common industry standard.

        Args:
            max_allowed_psi_score: the max threshold for the PSI score
            max_allowed_earth_movers_score: the max threshold for the Earth Mover's Distance score
            number_of_top_features_to_consider: the number of top features for which exceed the threshold will fail the
                condition.

        Returns:
            ConditionResult: False if any column has passed the max threshold, True otherwise
        """

        def condition(result: Dict) -> ConditionResult:
            if all(x['Importance'] is not None for x in result.values()):
                columns_to_consider = \
                    [col_name for col_name, fi in sorted(result.items(), key=lambda item: item[1]['Importance'],
                                                         reverse=True)]
            else:
                columns_to_consider = \
                    [col_name for col_name, fi in sorted(result.items(), key=lambda item: item[1]['Drift score'],
                                                         reverse=True)]
            columns_to_consider = columns_to_consider[:number_of_top_features_to_consider]
            not_passing_categorical_columns = [column for column, d in result.items() if
                                               d['Drift score'] > max_allowed_psi_score and d['Method'] == 'PSI'
                                               and column in columns_to_consider]
            not_passing_numeric_columns = [column for column, d in result.items() if
                                           d['Drift score'] > max_allowed_earth_movers_score
                                           and d['Method'] == "Earth Mover's Distance"
                                           and column in columns_to_consider]
            return_str = ''
            if not_passing_categorical_columns:
                return_str += f'Found categorical columns with PSI over {max_allowed_psi_score}: ' \
                              f'{", ".join(map(str, not_passing_categorical_columns))}\n'
            if not_passing_numeric_columns:
                return_str += f'Found numeric columns with Earth Mover\'s Distance over ' \
                              f'{max_allowed_earth_movers_score}: {", ".join(map(str, not_passing_numeric_columns))}'

            if return_str:
                return ConditionResult(False, return_str)
            else:
                return ConditionResult(True)

        return self.add_condition(f'PSI and Earth Mover\'s Distance cannot be greater than {max_allowed_psi_score} and '
                                  f'{max_allowed_earth_movers_score} respectively',
                                  condition)
