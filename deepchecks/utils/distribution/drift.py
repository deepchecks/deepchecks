# ----------------------------------------------------------------------------
# Copyright (C) 2021-2022 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""Common utilities for distribution checks."""

from typing import Tuple, Union, Hashable, Callable, Optional

from scipy.stats import wasserstein_distance
import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from deepchecks.utils.distribution.plot import drift_score_bar_traces, feature_distribution_traces
from deepchecks.utils.distribution.preprocessing import preprocess_2_cat_cols_to_same_bins
from deepchecks.core.errors import DeepchecksValueError, NotEnoughSamplesError
from deepchecks.utils.strings import format_percent

PSI_MIN_PERCENTAGE = 0.01

__all__ = ['calc_drift_and_plot']


def psi(expected_percents: np.ndarray, actual_percents: np.ndarray):
    """
    Calculate the PSI (Population Stability Index).

    See https://www.lexjansen.com/wuss/2017/47_Final_Paper_PDF.pdf

    Parameters
    ----------
    expected_percents: np.ndarray
        array of percentages of each value in the expected distribution.
    actual_percents: : np.ndarray
        array of percentages of each value in the actual distribution.
    Returns
    -------
    psi
        The PSI score

    """
    psi_value = 0
    for i in range(len(expected_percents)):
        # In order for the value not to diverge, we cap our min percentage value
        e_perc = max(expected_percents[i], PSI_MIN_PERCENTAGE)
        a_perc = max(actual_percents[i], PSI_MIN_PERCENTAGE)
        value = (e_perc - a_perc) * np.log(e_perc / a_perc)
        psi_value += value

    return psi_value


def earth_movers_distance(dist1: Union[np.ndarray, pd.Series], dist2: Union[np.ndarray, pd.Series]):
    """
    Calculate the Earth Movers Distance (Wasserstein distance).

    See https://en.wikipedia.org/wiki/Wasserstein_metric

    Function is for numerical data only.

    Parameters
    ----------
    dist1 : Union[np.ndarray, pd.Series]
        array of numberical values.
    dist2 : Union[np.ndarray, pd.Series]
        array of numberical values to compare dist1 to.
    Returns
    -------
    Any
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


def calc_drift_and_plot(train_column: pd.Series,
                        test_column: pd.Series,
                        value_name: Hashable,
                        column_type: str,
                        plot_title: Optional[str] = None,
                        max_num_categories_for_drift: int = 10,
                        max_num_categories_for_display: int = 10,
                        show_categories_by: str = 'train_largest',
                        min_samples: int = 10) -> Tuple[float, str, Callable]:
    """
    Calculate drift score per column.

    Parameters
    ----------
    train_column: pd.Series
        column from train dataset
    test_column: pd.Series
        same column from test dataset
    value_name: Hashable
        title of the x axis, if plot_title is None then also the title of the whole plot.
    column_type: str
        type of column (either "numerical" or "categorical")
    plot_title: str or None
        if None use value_name as title otherwise use this.
    max_num_categories_for_drift: int, default: 10
        Max number of allowed categories. If there are more, they are binned into an "Other" category.
    max_num_categories_for_display: int, default: 10
        Max number of categories to show in plot.
    show_categories_by: str, default: 'train_largest'
        Specify which categories to show for categorical features' graphs, as the number of shown categories is limited
        by max_num_categories_for_display. Possible values:
        - 'train_largest': Show the largest train categories.
        - 'test_largest': Show the largest test categories.
        - 'largest_difference': Show the largest difference between categories.
    min_samples: int, default: 10
        Minimum number of samples for each column in order to calculate draft
    Returns
    -------
    Tuple[float, str, Callable]
        drift score of the difference between the two columns' distributions (Earth movers distance for
        numerical, PSI for categorical)
        graph comparing the two distributions (density for numerical, stack bar for categorical)
    """
    train_dist = train_column.dropna().values.reshape(-1)
    test_dist = test_column.dropna().values.reshape(-1)

    if len(train_dist) < min_samples or len(test_dist) < min_samples:
        raise NotEnoughSamplesError(f'For drift need {min_samples} samples but got {len(train_dist)} for train '
                                    f'and {len(test_dist)} for test')

    if column_type == 'numerical':
        scorer_name = "Earth Mover's Distance"

        train_dist = train_dist.astype('float')
        test_dist = test_dist.astype('float')

        score = earth_movers_distance(dist1=train_dist, dist2=test_dist)
        bar_traces, bar_x_axis, bar_y_axis = drift_score_bar_traces(score)

        dist_traces, dist_x_axis, dist_y_axis = feature_distribution_traces(train_dist, test_dist, value_name)
    elif column_type == 'categorical':
        scorer_name = 'PSI'
        expected, actual, _ = \
            preprocess_2_cat_cols_to_same_bins(dist1=train_column, dist2=test_column,
                                               max_num_categories=max_num_categories_for_drift)
        expected_percents, actual_percents = expected / len(train_column), actual / len(test_column)
        score = psi(expected_percents=expected_percents, actual_percents=actual_percents)

        bar_traces, bar_x_axis, bar_y_axis = drift_score_bar_traces(score, bar_max=1)
        dist_traces, dist_x_axis, dist_y_axis = feature_distribution_traces(
            train_dist, test_dist, value_name, is_categorical=True, max_num_categories=max_num_categories_for_display,
            show_categories_by=show_categories_by
        )
    else:
        # Should never reach here
        raise DeepchecksValueError(f'Unsupported column type for drift: {column_type}')

    all_categories = list(set(train_column).union(set(test_column)))
    add_footnote = column_type == 'categorical' and len(all_categories) > max_num_categories_for_display

    if not add_footnote:
        fig = make_subplots(rows=2, cols=1, vertical_spacing=0.2, shared_yaxes=False, shared_xaxes=False,
                            row_heights=[0.1, 0.9],
                            subplot_titles=[f'Drift Score ({scorer_name})', 'Distribution Plot'])
    else:
        fig = make_subplots(rows=3, cols=1, vertical_spacing=0.2, shared_yaxes=False, shared_xaxes=False,
                            row_heights=[0.1, 0.8, 0.1],
                            subplot_titles=[f'Drift Score ({scorer_name})', 'Distribution Plot'])

    fig.add_traces(bar_traces, rows=[1] * len(bar_traces), cols=[1] * len(bar_traces))
    fig.add_traces(dist_traces, rows=[2] * len(dist_traces), cols=[1] * len(dist_traces))

    if add_footnote:
        param_to_print_dict = {
            'train_largest': 'largest categories (by train)',
            'test_largest': 'largest categories (by test)',
            'largest_difference': 'largest difference between categories'
        }
        train_data_percents = dist_traces[0].y.sum()
        test_data_percents = dist_traces[1].y.sum()

        fig.add_annotation(
            x=0, y=-0.2, showarrow=False, xref='paper', yref='paper', xanchor='left',
            text=f'* Showing the top {max_num_categories_for_display} {param_to_print_dict[show_categories_by]} out of '
                 f'total {len(all_categories)} categories.'
                 f'<br>Shown data is {format_percent(train_data_percents)} of train data and '
                 f'{format_percent(test_data_percents)} of test data.'
        )

    if not plot_title:
        plot_title = value_name

    shared_layout = go.Layout(
        xaxis=bar_x_axis,
        yaxis=bar_y_axis,
        xaxis2=dist_x_axis,
        yaxis2=dist_y_axis,
        legend=dict(
            title='Dataset',
            yanchor='top',
            y=0.6),
        width=700,
        height=400,
        title=dict(text=plot_title, x=0.5, xanchor='center'),
        bargroupgap=0
    )

    fig.update_layout(shared_layout)

    return score, scorer_name, fig
