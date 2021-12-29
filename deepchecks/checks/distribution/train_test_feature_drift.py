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

from collections import OrderedDict
from typing import Union, Tuple, List, Dict, Callable, Optional

import pandas as pd

from deepchecks import Dataset, CheckResult, TrainTestBaseCheck, ConditionResult
from deepchecks.checks.distribution.plot import drift_score_bar_traces, feature_distribution_traces
from deepchecks.checks.distribution.dist_utils import preprocess_for_psi, earth_movers_distance, psi
from deepchecks.utils.features import calculate_feature_importance_or_none
from deepchecks.utils.typing import Hashable
from deepchecks.errors import DeepchecksValueError
import plotly.graph_objects as go
from plotly.subplots import make_subplots


__all__ = ['TrainTestFeatureDrift']


class TrainTestFeatureDrift(TrainTestBaseCheck):
    """
    Calculate drift between train dataset and test dataset per feature, using statistical measures.

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
            they are binned into an "Other" category. If max_num_categories=None, there is no limit. This limit applies
            for both drift calculation and for distribution plots.
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
            train_dataset (Dataset): The training dataset object.
            test_dataset (Dataset): The test dataset object.
            model: A scikit-learn-compatible fitted estimator instance

        Returns:
            CheckResult:
                value: dictionary of column name to drift score.
                display: distribution graph for each column, comparing the train and test distributions.

        Raises:
            DeepchecksValueError: If the object is not a Dataset or DataFrame instance
        """
        feature_importances = calculate_feature_importance_or_none(model, train_dataset)
        return self._calc_drift(train_dataset, test_dataset, feature_importances)

    def _calc_drift(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        feature_importances: Optional[pd.Series] = None
    ) -> CheckResult:
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

        train_dataset = train_dataset.select(self.columns, self.ignore_columns)
        test_dataset = test_dataset.select(self.columns, self.ignore_columns)

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

        if feature_importances is not None:
            fi_rank_series = feature_importances.rank(method='first', ascending=False)
            fi_rank = fi_rank_series[column_name]
            plot_title = f'{column_name} (#{int(fi_rank)} in FI)'
        else:
            plot_title = column_name

        if column_type == 'numerical':
            scorer_name = "Earth Mover's Distance"
            score = earth_movers_distance(dist1=train_column.astype('float'), dist2=test_column.astype('float'))
            bar_stop = max(0.4, score + 0.1)

            score_bar = drift_score_bar_traces(score)

            traces, xaxis_layout, yaxis_layout = feature_distribution_traces(train_column,
                                                                             test_dist)

        elif column_type == 'categorical':
            scorer_name = 'PSI'
            expected_percents, actual_percents, _ = \
                preprocess_for_psi(dist1=train_dist, dist2=test_dist, max_num_categories=self.max_num_categories)
            score = psi(expected_percents=expected_percents, actual_percents=actual_percents)
            bar_stop = max(0.4, score + 0.1)

            score_bar = drift_score_bar_traces(score)

            traces, xaxis_layout, yaxis_layout = feature_distribution_traces(train_dist,
                                                                             test_dist,
                                                                             is_categorical=True,
                                                                             max_num_categories=self.max_num_categories)

        fig = make_subplots(rows=2, cols=1, vertical_spacing=0.4, shared_yaxes=False, shared_xaxes=False,
                            row_heights=[0.1, 0.9],
                            subplot_titles=['Drift Score - ' + scorer_name, plot_title])

        fig.add_traces(score_bar, rows=[1] * len(score_bar), cols=[1] * len(score_bar))
        fig.add_traces(traces, rows=[2] * len(traces), cols=[1] * len(traces))

        shared_layout = go.Layout(
            xaxis=dict(
                showgrid=False,
                gridcolor='black',
                linecolor='black',
                range=[0, bar_stop],
                dtick=0.05,
                title='drift score'
            ),
            yaxis=dict(
                showgrid=False,
                showline=False,
                showticklabels=False,
                zeroline=False,
            ),
            xaxis2=xaxis_layout,
            yaxis2=yaxis_layout,
            legend=dict(
                title='Dataset',
                yanchor='top',
                y=0.7,
                xanchor='left',
                x=0.85),
            width=700,
            height=400
        )

        fig.update_layout(shared_layout)

        return score, scorer_name, fig

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
