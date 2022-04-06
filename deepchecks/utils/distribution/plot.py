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
"""A module containing utils for plotting distributions."""
from numbers import Number
from functools import cmp_to_key

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import plotly.graph_objs as go

__all__ = ['feature_distribution_traces', 'drift_score_bar_traces', 'get_density']

from typing import List, Dict, Tuple

from deepchecks.utils.distribution.preprocessing import preprocess_2_cat_cols_to_same_bins
from deepchecks.utils.plot import colors, hex_to_rgba
from deepchecks.utils.dataframes import un_numpy


def get_density(data, xs) -> np.ndarray:
    """Get gaussian kde density to plot.

    Parameters
    ----------
    data
        The data used to compute the pdf function.
    xs : iterable
        List of x values to plot the computed pdf for.
    Returns
    -------
    np.array
        The computed pdf values at the points xs.
    """
    # Is only single value adds noise, otherwise there is singular matrix error
    if len(np.unique(data)) == 1:
        data = data + np.random.normal(scale=10 * np.finfo(np.float32).eps, size=len(data))
    density = gaussian_kde(data)
    density.covariance_factor = lambda: .25
    # pylint: disable=protected-access
    density._compute_covariance()
    return density(xs)


def drift_score_bar_traces(drift_score: float, bar_max: float = None) -> Tuple[List[go.Bar], Dict, Dict]:
    """Create a traffic light bar traces for drift score.

    Parameters
    ----------
    drift_score : float
        Drift score
    bar_max : float , default: None
        Maximum value for the bar
    Returns
    -------
    Tuple[List[go.Bar], Dict, Dict]
        list of plotly bar traces.
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
            x=[min(drift_score, range_tuple[1]) - range_tuple[0]],
            y=['Drift Score'],
            orientation='h',
            marker=dict(
                color=color,
            ),
            offsetgroup=0,
            base=range_tuple[0],
            showlegend=False

        ))

    bar_stop = max(0.4, drift_score + 0.1)
    if bar_max:
        bar_stop = min(bar_stop, bar_max)
    xaxis = dict(
        showgrid=False,
        gridcolor='black',
        linecolor='black',
        range=[0, bar_stop],
        dtick=0.05,
        fixedrange=True
    )
    yaxis = dict(
        showgrid=False,
        showline=False,
        showticklabels=False,
        zeroline=False,
        color='black',
        fixedrange=True
    )

    return bars, xaxis, yaxis


def feature_distribution_traces(train_column,
                                test_column,
                                column_name,
                                is_categorical: bool = False,
                                max_num_categories: int = 10,
                                quantile_cut: float = 0.02) -> Tuple[List[go.Trace], Dict, Dict]:
    """Create traces for comparison between train and test column.

    Parameters
    ----------
    train_column
        Train data used to trace distribution.
    test_column
        Test data used to trace distribution.
    column_name
        The name of the column values on the x axis.
    is_categorical : bool , default: False
        State if column is categorical.
    max_num_categories : int , default: 10
        Maximum number of categories to show in plot (default: 10).
    quantile_cut : float , default: 0.02
        in which quantile to cut the edges of the plot
    Returns
    -------
    List[Union[go.Bar, go.Scatter]]
        list of plotly traces.
    Dict
        layout of x axis
    Dict
        layout of y axis
    """
    if is_categorical:
        expected_percents, actual_percents, categories_list = \
            preprocess_2_cat_cols_to_same_bins(dist1=train_column, dist2=test_column,
                                               max_num_categories=max_num_categories)
        # fixes plotly widget bug with numpy values by converting them to native values
        # https://github.com/plotly/plotly.py/issues/3470
        categories_list = [un_numpy(cat) for cat in categories_list]
        cat_df = pd.DataFrame({'Train dataset': expected_percents, 'Test dataset': actual_percents},
                              index=categories_list)

        # Creating sorting function which works on both numbers and strings
        def sort_int_and_strings(a, b):
            # If both numbers or both same type using regular operator
            if a.__class__ == b.__class__ or (isinstance(a, Number) and isinstance(b, Number)):
                return -1 if a < b else 1
            # Sort numbers before strings
            return -1 if isinstance(a, Number) else 1
        cat_df = cat_df.reindex(sorted(cat_df.index, key=cmp_to_key(sort_int_and_strings)))

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

        max_y = max(*expected_percents, *actual_percents)
        y_lim = 1 if max_y > 0.5 else max_y * 1.1

        xaxis_layout = dict(type='category')
        yaxis_layout = dict(fixedrange=True,
                            range=(0, y_lim),
                            title='Percentage')

    else:
        is_train_single_value = train_column.min() == train_column.max()
        is_test_single_value = test_column.min() == test_column.max()
        # If both columns are single value, then return a table instead of graph
        if is_train_single_value and is_test_single_value:
            table = go.Table(header=dict(values=['Train Dataset Value', 'Test Dataset Value']),
                             cells=dict(values=[[train_column[0]], [test_column[0]]]))
            return [table], {}, {}

        x_range = (min(train_column.min(), test_column.min()), max(train_column.max(), test_column.max()))
        x_range_to_show = (
            min(np.quantile(train_column, quantile_cut), np.quantile(test_column, quantile_cut)),
            max(np.quantile(train_column, 1 - quantile_cut), np.quantile(test_column, 1 - quantile_cut))
        )
        # Heuristically take points on x-axis to show on the plot
        # The intuition is the graph will look "smooth" wherever we will zoom it
        xs = sorted(np.concatenate((
            np.linspace(x_range[0], x_range[1], 50),
            np.quantile(train_column, q=np.arange(0.02, 1, 0.02)),
            np.quantile(test_column, q=np.arange(0.02, 1, 0.02))
        )))

        train_density = get_density(train_column, xs)
        test_density = get_density(test_column, xs)

        traces = []
        if is_train_single_value:
            traces.append(go.Scatter(
                x=[train_column.min()] * 2,
                # Draw the line a bit higher than the max value of test density
                y=[0, np.max(test_density) * 1.1],
                line=dict(
                    color=hex_to_rgba(colors['Train'], 0.7),
                ),
                name='Train Dataset',
                mode='lines'
            ))
        else:
            traces.append(go.Scatter(x=xs, y=train_density, fill='tozeroy', name='Train Dataset',
                          line_color=colors['Train']))

        if is_test_single_value:
            traces.append(go.Scatter(
                x=[test_column.min()] * 2,
                # Draw the line a bit higher than the max value of train density
                y=[0, np.max(train_density) * 1.1],
                line=dict(
                    color=hex_to_rgba(colors['Test'], 0.7),
                ),
                name='Test Dataset',
                mode='lines'
            ))
        else:
            traces.append(go.Scatter(x=xs, y=test_density, fill='tozeroy', name='Test Dataset',
                          line_color=colors['Test']))

        xaxis_layout = dict(fixedrange=False,
                            range=x_range_to_show,
                            title=column_name)
        yaxis_layout = dict(title='Probability Density', fixedrange=True)

    return traces, xaxis_layout, yaxis_layout
