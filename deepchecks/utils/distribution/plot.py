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

from deepchecks.core.errors import DeepchecksValueError
from deepchecks.utils.distribution.preprocessing import preprocess_2_cat_cols_to_same_bins
from deepchecks.utils.plot import colors
from deepchecks.utils.dataframes import un_numpy

# For numerical plots, below this number of unique values we draw bar plots, else KDE
MAX_NUMERICAL_UNIQUE_FOR_BARS = 20
# For numerical plots, where the total unique is above MAX_NUMERICAL_UNIQUE_FOR_BARS, if any of the single
# datasets have unique values above this number, we draw KDE, else we draw bar plots. Should be less than half of
# MAX_NUMERICAL_UNIQUE_FOR_BARS
MAX_NUMERICAL_UNIQUES_FOR_SINGLE_DIST_BARS = 5


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
                                show_categories_by: str = 'train_largest',
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
    show_categories_by: str, default: 'train_largest'
        Specify which categories to show for categorical features' graphs, as the number of shown categories is limited
        by max_num_categories_for_display. Possible values:
        - 'train_largest': Show the largest train categories.
        - 'test_largest': Show the largest test categories.
        - 'largest_difference': Show the largest difference between categories.
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
    Dict
        general layout
    """
    if is_categorical:
        traces, y_layout = _create_distribution_bar_graphs(train_column, test_column, max_num_categories,
                                                           show_categories_by)
        xaxis_layout = dict(type='category')
        return traces, xaxis_layout, y_layout
    else:
        train_uniques, train_uniques_counts = np.unique(train_column, return_counts=True)
        test_uniques, test_uniques_counts = np.unique(test_column, return_counts=True)
        x_range = (min(train_column.min(), test_column.min()), max(train_column.max(), test_column.max()))

        # If there are less than 20 total unique values, draw bar graph
        train_test_uniques = np.unique(np.concatenate([train_uniques, test_uniques]))
        if train_test_uniques.size < MAX_NUMERICAL_UNIQUE_FOR_BARS:
            traces, y_layout = _create_distribution_bar_graphs(train_column, test_column, 20, show_categories_by)
            # In case of single value widen the range, else plotly draw the bars really wide.
            if x_range[0] == x_range[1]:
                x_range = (x_range[0] - 5, x_range[0] + 5)
            # In case of multi values still widen the range, else plotly hide the bars in the edges.
            else:
                x_range = None
            xaxis_layout = dict(ticks='outside', tickmode='array', tickvals=train_test_uniques, range=x_range)
            return traces, xaxis_layout, y_layout

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
        bars_width = (x_range_to_show[1] - x_range_to_show[0]) / 100

        traces = []
        if train_uniques.size <= MAX_NUMERICAL_UNIQUES_FOR_SINGLE_DIST_BARS:
            traces.append(go.Bar(
                x=train_uniques,
                y=_create_bars_data_for_mixed_kde_plot(train_uniques_counts, np.max(test_density)),
                width=[bars_width] * train_uniques.size,
                marker=dict(
                    color=colors['Train'],
                ),
                name='Train Dataset',
            ))
        else:
            traces.append(go.Scatter(x=xs, y=train_density, fill='tozeroy', name='Train Dataset',
                                     line_color=colors['Train']))

        if test_uniques.size <= MAX_NUMERICAL_UNIQUES_FOR_SINGLE_DIST_BARS:
            traces.append(go.Bar(
                x=test_uniques,
                y=_create_bars_data_for_mixed_kde_plot(test_uniques_counts, np.max(train_density)),
                width=[bars_width] * test_uniques.size,
                marker=dict(
                    color=colors['Test']
                ),
                name='Test Dataset',
            ))
        else:
            traces.append(go.Scatter(x=xs, y=test_density, fill='tozeroy', name='Test Dataset',
                                     line_color=colors['Test']))

        xaxis_layout = dict(fixedrange=False,
                            range=x_range_to_show,
                            title=column_name)
        yaxis_layout = dict(title='Probability Density', fixedrange=True)
        return traces, xaxis_layout, yaxis_layout


def _create_bars_data_for_mixed_kde_plot(counts: np.ndarray, max_kde_value: float):
    """When showing a mixed KDE and bar plot, we want the bars to be on the same scale of y-values as the KDE values, \
    so we normalize the counts to sum to 4 times the max KDE value."""
    normalize_factor = 4 * max_kde_value / np.sum(counts)
    return counts * normalize_factor


def _create_distribution_bar_graphs(train_column, test_column, max_num_categories: int, show_categories_by: str):
    expected, actual, categories_list = \
        preprocess_2_cat_cols_to_same_bins(dist1=train_column, dist2=test_column)

    expected_percents, actual_percents = expected / len(train_column), actual / len(test_column)

    # Get sorting lambda function according to the parameter show_categories_by:
    if show_categories_by == 'train_largest':
        sort_func = lambda tup: tup[0]
    elif show_categories_by == 'test_largest':
        sort_func = lambda tup: tup[1]
    elif show_categories_by == 'largest_difference':
        sort_func = lambda tup: np.abs(tup[0] - tup[1])
    else:
        raise DeepchecksValueError(f'show_categories_by must be either "train_largest", "test_largest" '
                                   f'or "largest_difference", instead got: {show_categories_by}')

    # Sort the lists together according to the parameter show_categories_by (done by sorting zip and then using it again
    # to return the lists to the original 3 separate ones).
    # Afterwards, leave only the first max_num_categories values in each list.
    expected_percents, actual_percents, categories_list = zip(
        *list(sorted(zip(expected_percents, actual_percents, categories_list), key=sort_func, reverse=True))[
         :max_num_categories])

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

    yaxis_layout = dict(fixedrange=True,
                        range=(0, y_lim),
                        title='Frequency')

    return traces, yaxis_layout
