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
import typing as t
from functools import cmp_to_key
from numbers import Number

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from scipy.stats import gaussian_kde
from typing_extensions import Literal as L

__all__ = ['feature_distribution_traces', 'drift_score_bar_traces', 'get_density']

from typing import Dict, List, Tuple

from deepchecks.core.errors import DeepchecksValueError
from deepchecks.utils.dataframes import un_numpy
from deepchecks.utils.distribution.preprocessing import preprocess_2_cat_cols_to_same_bins
from deepchecks.utils.plot import colors

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
        autorange=True,
        rangemode='normal',
        fixedrange=True
    )

    return bars, xaxis, yaxis


CategoriesSortingKind = t.Union[L['train_largest'], L['test_largest'], L['largest_difference']]  # noqa: F821


def feature_distribution_traces(
    train_column: t.Union[np.ndarray, pd.Series],
    test_column: t.Union[np.ndarray, pd.Series],
    column_name: str,
    is_categorical: bool = False,
    max_num_categories: int = 10,
    show_categories_by: CategoriesSortingKind = 'largest_difference',
    quantile_cut: float = 0.02
) -> Tuple[List[go.Trace], Dict, Dict]:
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
    show_categories_by: str, default: 'largest_difference'
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
        n_of_categories = len(set(train_column).union(test_column))
        range_max = (
            max_num_categories
            if n_of_categories > max_num_categories
            else n_of_categories
        )
        traces, y_layout = _create_distribution_bar_graphs(
            train_column,
            test_column,
            max_num_categories,
            show_categories_by
        )
        xaxis_layout = dict(
            type='category',
            # NOTE:
            # the range, in this case, is needed to fix a problem with
            # too wide bars when there are only one or two of them`s on
            # the plot, plus it also centralizes them`s on the plot
            # The min value of the range (range(min. max)) is bigger because
            # otherwise bars will not be centralized on the plot, they will
            # appear on the left part of the plot (that is probably because of zero)
            range=(-3, range_max + 2)
        )
        return traces, xaxis_layout, y_layout
    else:
        train_uniques, train_uniques_counts = np.unique(train_column, return_counts=True)
        test_uniques, test_uniques_counts = np.unique(test_column, return_counts=True)

        x_range = (
            min(train_column.min(), test_column.min()),
            max(train_column.max(), test_column.max())
        )

        # If there are less than 20 total unique values, draw bar graph
        train_test_uniques = np.unique(np.concatenate([train_uniques, test_uniques]))
        if train_test_uniques.size < MAX_NUMERICAL_UNIQUE_FOR_BARS:
            traces, y_layout = _create_distribution_bar_graphs(train_column, test_column, 20, show_categories_by)
            x_range = (x_range[0] - 5, x_range[1] + 5)
            xaxis_layout = dict(ticks='outside', tickmode='array', tickvals=train_test_uniques, range=x_range)
            return traces, xaxis_layout, y_layout

        x_range_to_show = (
            min(np.quantile(train_column, quantile_cut), np.quantile(test_column, quantile_cut)),
            max(np.quantile(train_column, 1 - quantile_cut), np.quantile(test_column, 1 - quantile_cut))
        )
        # Heuristically take points on x-axis to show on the plot
        # The intuition is the graph will look "smooth" wherever we will zoom it
        # Also takes mean and median values in order to plot it later accurately
        mean_train_column = np.mean(train_column)
        mean_test_column = np.mean(test_column)
        median_train_column = np.median(train_column)
        median_test_column = np.median(test_column)
        xs = sorted(np.concatenate((
            np.linspace(x_range[0], x_range[1], 50),
            np.quantile(train_column, q=np.arange(0.02, 1, 0.02)),
            np.quantile(test_column, q=np.arange(0.02, 1, 0.02)),
            [mean_train_column, mean_test_column, median_train_column, median_test_column]
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
                marker=dict(color=colors['Train']),
                name='Train Dataset',
            ))
        else:
            traces.extend(_create_distribution_scatter_plot(xs, train_density, mean_train_column, median_train_column,
                                                            is_train=True))

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
            traces.extend(_create_distribution_scatter_plot(xs, test_density, mean_test_column, median_test_column,
                                                            is_train=False))

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


def _create_distribution_scatter_plot(xs, ys, mean, median, is_train):
    traces = []
    name = 'Train' if is_train else 'Test'
    traces.append(go.Scatter(x=xs, y=ys, fill='tozeroy', name=f'{name} Dataset',
                             line_color=colors[name], line_shape='spline'))
    y_mean_index = np.argmax(xs == mean)
    traces.append(go.Scatter(x=[mean, mean], y=[0, ys[y_mean_index]], name=f'{name} Mean',
                             line=dict(color=colors[name], dash='dash'), mode='lines+markers'))
    y_median_index = np.argmax(xs == median)
    traces.append(go.Scatter(x=[median, median], y=[0, ys[y_median_index]], name=f'{name} Median',
                             line=dict(color=colors[name]), mode='lines'))
    return traces


def _create_distribution_bar_graphs(
    train_column: t.Union[np.ndarray, pd.Series],
    test_column: t.Union[np.ndarray, pd.Series],
    max_num_categories: int,
    show_categories_by: CategoriesSortingKind
) -> t.Tuple[t.Any, t.Any]:
    """
    Create distribution bar graphs.

    Returns
    -------
    Tuple[Any, Any]:
        a tuple instance with figues traces, yaxis layout
    """
    expected, actual, categories_list = preprocess_2_cat_cols_to_same_bins(
        dist1=train_column,
        dist2=test_column
    )

    expected_percents, actual_percents = expected / len(train_column), actual / len(test_column)

    if show_categories_by == 'train_largest':
        sort_func = lambda tup: tup[0]
    elif show_categories_by == 'test_largest':
        sort_func = lambda tup: tup[1]
    elif show_categories_by == 'largest_difference':
        sort_func = lambda tup: np.abs(tup[0] - tup[1])
    else:
        raise DeepchecksValueError(
            'show_categories_by must be either "train_largest", "test_largest" '
            f'or "largest_difference", instead got: {show_categories_by}'
        )

    # Sort the lists together according to the parameter show_categories_by (done by sorting zip and then using it again
    # to return the lists to the original 3 separate ones).
    # Afterwards, leave only the first max_num_categories values in each list.
    distribution = sorted(
        zip(expected_percents, actual_percents, categories_list),
        key=sort_func,
        reverse=True
    )
    expected_percents, actual_percents, categories_list = zip(
        *distribution[:max_num_categories]
    )

    # fixes plotly widget bug with numpy values by converting them to native values
    # https://github.com/plotly/plotly.py/issues/3470
    cat_df = pd.DataFrame(
        {'Train dataset': expected_percents, 'Test dataset': actual_percents},
        index=[un_numpy(cat) for cat in categories_list]
    )

    # Creating sorting function which works on both numbers and strings
    def sort_int_and_strings(a, b):
        # If both numbers or both same type using regular operator
        if type(a) is type(b) or (isinstance(a, Number) and isinstance(b, Number)):
            return -1 if a < b else 1
        # Sort numbers before strings
        return -1 if isinstance(a, Number) else 1

    cat_df = cat_df.reindex(sorted(cat_df.index, key=cmp_to_key(sort_int_and_strings)))

    traces = [
        go.Bar(
            x=cat_df.index,
            y=cat_df['Train dataset'],
            marker=dict(color=colors['Train']),
            name='Train Dataset',
        ),
        go.Bar(
            x=cat_df.index,
            y=cat_df['Test dataset'],
            marker=dict(color=colors['Test']),
            name='Test Dataset',
        )
    ]

    yaxis_layout = dict(
        fixedrange=True,
        autorange=True,
        rangemode='normal',
        title='Frequency'
    )

    return traces, yaxis_layout
