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
"""A module containing utils for plotting distributions."""
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import plotly.graph_objs as go

__all__ = ['feature_distribution_traces', 'drift_score_bar_traces', 'get_density']

from typing import List, Union, Dict

from deepchecks.checks.distribution.preprocessing import preprocess_2_cat_cols_to_same_bins
from deepchecks.utils.plot import colors


def get_density(data, xs) -> np.ndarray:
    """Get gaussian kde density to plot.

    Args:
        data (): The data used to compute the pdf function.
        xs (iterable): List of x values to plot the computed pdf for.

    Returns:
        np.array: The computed pdf values at the points xs.
    """
    density = gaussian_kde(data)
    density.covariance_factor = lambda: .25
    # pylint: disable=protected-access
    density._compute_covariance()
    return density(xs)


def drift_score_bar_traces(drift_score: float) -> List[go.Bar]:
    """Create a traffic light bar traces for drift score.

    Args:
        drift_score (float): Drift score

    Returns:
        List[go.Bar]: list of plotly bar traces.
    """
    traffic_light_colors = [((0, 0.1), '#01B8AA'),
                            ((0.1, 0.2), '#F2C80F'),
                            ((0.2, 0.3), '#FE9666'),
                            ((0.3, 1), '#FD625E')
                            ]

    bars = []

    for range_tuple, color in traffic_light_colors:
        if drift_score < range_tuple[0]:
            break

        bars.append(go.Bar(
            x=[min(drift_score, range_tuple[1]) - range_tuple[0]], y=['Drift Score'],
            orientation='h',
            marker=dict(
                color=color,
            ),
            offsetgroup=0,
            base=range_tuple[0],
            showlegend=False

        ))

    return bars


def feature_distribution_traces(train_column,
                                test_column,
                                is_categorical: bool = False,
                                max_num_categories: int = 10) -> [List[Union[go.Bar, go.Scatter]], Dict, Dict]:
    """Create traces for comparison between train and test column.

    Args:
        train_column (): Train data used to trace distribution.
        test_column (): Test data used to trace distribution.
        is_categorical (bool): State if column is categorical (default: False).
        max_num_categories (int): Maximum number of categories to show in plot (default: 10).

    Returns:
        List[Union[go.Bar, go.Scatter]]: list of plotly traces.
        Dict: layout of x axis
        Dict: layout of y axis
    """
    if is_categorical:
        expected_percents, actual_percents, categories_list = \
            preprocess_2_cat_cols_to_same_bins(dist1=train_column, dist2=test_column,
                                               max_num_categories=max_num_categories)
        cat_df = pd.DataFrame({'Train dataset': expected_percents, 'Test dataset': actual_percents},
                              index=categories_list)
        train_bar = go.Bar(
            x=cat_df.index,
            y=cat_df['Train dataset'],
            marker=dict(
                color=colors['Train'],
            ),
            name='Train Dataset',
        )

        test_bar = go.Bar(
            x=cat_df.index,
            y=cat_df['Test dataset'],
            marker=dict(
                color=colors['Test'],
            ),
            name='Test Dataset',
        )

        traces = [train_bar, test_bar]

        xaxis_layout = dict(type='category')
        yaxis_layout = dict(fixedrange=True,
                            range=(0, 1),
                            title='Percentage')

    else:
        x_range = (min(train_column.min(), test_column.min()), max(train_column.max(), test_column.max()))
        xs = np.linspace(x_range[0], x_range[1], 40)

        traces = [go.Scatter(x=xs, y=get_density(train_column, xs), fill='tozeroy', name='Train Dataset',
                             line_color=colors['Train']),
                  go.Scatter(x=xs, y=get_density(test_column, xs), fill='tozeroy', name='Test Dataset',
                             line_color=colors['Test'])]

        xaxis_layout = dict(fixedrange=True,
                            range=x_range,
                            title='Distribution')
        yaxis_layout = dict(title='Probability Density')

    return traces, xaxis_layout, yaxis_layout
