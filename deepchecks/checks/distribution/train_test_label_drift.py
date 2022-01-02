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
"""Module contains Train Test label Drift check."""

from typing import Tuple, Dict, Callable

import pandas as pd

from deepchecks import Dataset, CheckResult, TrainTestBaseCheck, ConditionResult
from deepchecks.checks.distribution.plot import drift_score_bar_traces, feature_distribution_traces
from deepchecks.checks.distribution.dist_utils import preprocess_for_psi, earth_movers_distance, psi
from deepchecks.utils.typing import Hashable
import plotly.graph_objects as go
from plotly.subplots import make_subplots

__all__ = ['TrainTestLabelDrift']


class TrainTestLabelDrift(TrainTestBaseCheck):
    """
    Calculate label drift between train dataset and test dataset, using statistical measures.

    Check calculates a drift score for the label in test dataset, by comparing its distribution to the train
    dataset.
    For numerical columns, we use the Earth Movers Distance.
    See https://www.lexjansen.com/wuss/2017/47_Final_Paper_PDF.pdf
    For categorical columns, we use the Population Stability Index (PSI).
    See https://en.wikipedia.org/wiki/Wasserstein_metric.


    Args:
        max_num_categories (int):
            Only for categorical columns. Max number of allowed categories. If there are more,
            they are binned into an "Other" category. If max_num_categories=None, there is no limit. This limit applies
            for both drift calculation and for distribution plots.
    """

    def __init__(
            self,
            max_num_categories: int = 10
    ):
        super().__init__()
        self.max_num_categories = max_num_categories

    def run(self, train_dataset, test_dataset, model=None) -> CheckResult:
        """Run check.

        Args:
            train_dataset (Dataset): The training dataset object.
            test_dataset (Dataset): The test dataset object.
            model: not used in this check.

        Returns:
            CheckResult:
                value: dictionary of column name to drift score.
                display: distribution graph for each column, comparing the train and test distributions.

        Raises:
            DeepchecksValueError: If the object is not a Dataset or DataFrame instance
        """
        return self._calc_drift(train_dataset, test_dataset)

    def _calc_drift(
            self,
            train_dataset: Dataset,
            test_dataset: Dataset,
    ) -> CheckResult:
        """
        Calculate drift for all columns.

        Args:
            train_dataset (Dataset): The training dataset object. Must contain a label.
            test_dataset (Dataset): The test dataset object. Must contain a label.

        Returns:
            CheckResult:
                value: drift score.
                display: label distribution graph, comparing the train and test distributions.
        """
        train_dataset = Dataset.validate_dataset(train_dataset)
        test_dataset = Dataset.validate_dataset(test_dataset)
        train_dataset.validate_label()
        test_dataset.validate_label()

        drift_score, method, display = self._calc_drift_per_column(
            train_column=train_dataset.label_col,
            test_column=test_dataset.label_col,
            column_name=train_dataset.label_name,
            column_type='categorical' if train_dataset.label_type == 'classification_label' else 'numerical',
        )

        headnote = """<span>
            The Drift score is a measure for the difference between two distributions, in this check - the test
            and train distributions.<br> The check shows the drift score and distributions for the label.
        </span>"""

        displays = [headnote, display]
        values_dict = {'Drift score': drift_score, 'Method': method}

        return CheckResult(value=values_dict, display=displays, header='Train Test Label Drift')

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

            traces, xaxis_layout, yaxis_layout = feature_distribution_traces(train_dist,
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
                y=0.6),
            width=700,
            height=400
        )

        fig.update_layout(shared_layout)

        return score, scorer_name, fig

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
            drift_score = result['Drift score']
            method = result['Method']
            has_failed = (drift_score > max_allowed_psi_score and method == 'PSI') or \
                         (drift_score > max_allowed_earth_movers_score and method == "Earth Mover's Distance")

            if method == 'PSI' and has_failed:
                return_str = f'Label has PSI over {max_allowed_psi_score} - Drift score is {drift_score:.2f}'
                return ConditionResult(False, return_str)
            elif method == "Earth Mover's Distance" and has_failed:
                return_str = f'Label has Earth Mover\'s Distance over {max_allowed_earth_movers_score} - ' \
                             f'Drift score is {drift_score:.2f}'
                return ConditionResult(False, return_str)

            return ConditionResult(True)

        return self.add_condition(f'PSI and Earth Mover\'s Distance for label drift cannot be greater than '
                                  f'{max_allowed_psi_score} or {max_allowed_earth_movers_score} respectively',
                                  condition)
