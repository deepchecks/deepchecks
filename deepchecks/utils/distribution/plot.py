# ----------------------------------------------------------------------------
# Copyright (C) 2021-2023 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""A module containing utils for plotting distributions."""
from functools import cmp_to_key
from numbers import Number
from typing import Any, Dict, List, Tuple, Union, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.basedatatypes import BaseTraceType
from scipy.stats import gaussian_kde
from typing_extensions import Literal as L

from deepchecks.core.errors import DeepchecksValueError
from deepchecks.nlp.utils.text_utils import break_to_lines_and_trim
from deepchecks.utils.dataframes import un_numpy
from deepchecks.utils.distribution.preprocessing import preprocess_2_cat_cols_to_same_bins
from deepchecks.utils.plot import DEFAULT_DATASET_NAMES, colors, common_and_outlier_colors

__all__ = ['feature_distribution_traces', 'drift_score_bar_traces', 'get_density', 'CategoriesSortingKind']

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


CategoriesSortingKind = L['train_largest', 'test_largest', 'largest_difference']  # noqa: F821


def feature_distribution_traces(
        train_column: Union[np.ndarray, pd.Series],
        test_column: Union[np.ndarray, pd.Series],
        column_name: str,
        is_categorical: bool = False,
        max_num_categories: int = 10,
        show_categories_by: CategoriesSortingKind = 'largest_difference',
        quantile_cut: float = 0.02,
        dataset_names: Tuple[str] = DEFAULT_DATASET_NAMES
) -> Tuple[List[BaseTraceType], Dict, Dict]:
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
        In which quantile to cut the edges of the plot.
    dataset_names: tuple, default: DEFAULT_DATASET_NAMES
        The names to show in the display for the first and second datasets.

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
        traces, y_layout = _create_distribution_bar_graphs(train_column, test_column,
                                                           max_num_categories, show_categories_by,
                                                           dataset_names=dataset_names)

        # NOTE:
        # the range, in this case, is needed to fix a problem with
        # too wide bars when there are only one or two of them`s on
        # the plot, plus it also centralizes them`s on the plot
        # The min value of the range (range(min. max)) is bigger because
        # otherwise bars will not be centralized on the plot, they will
        # appear on the left part of the plot (that is probably because of zero)
        range_max = max_num_categories if len(set(train_column).union(test_column)) > max_num_categories \
            else len(set(train_column).union(test_column))
        xaxis_layout = dict(type='category', range=(-3, range_max + 2))
        return traces, xaxis_layout, y_layout
    else:
        train_uniques, train_uniques_counts = np.unique(train_column, return_counts=True)
        test_uniques, test_uniques_counts = np.unique(test_column, return_counts=True)

        x_range = (
            min(train_column.min(), test_column.min()),
            max(train_column.max(), test_column.max())
        )
        x_width = x_range[1] - x_range[0]

        # If there are less than 20 total unique values, draw bar graph
        train_test_uniques = np.unique(np.concatenate([train_uniques, test_uniques]))
        if train_test_uniques.size < MAX_NUMERICAL_UNIQUE_FOR_BARS:
            traces, y_layout = _create_distribution_bar_graphs(train_column, test_column, 20, show_categories_by,
                                                               dataset_names=dataset_names)
            x_range = (x_range[0] - x_width * 0.2, x_range[1] + x_width * 0.2)
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

        traces: List[go.BaseTraceType] = []
        if train_uniques.size <= MAX_NUMERICAL_UNIQUES_FOR_SINGLE_DIST_BARS:
            traces.append(go.Bar(
                x=train_uniques,
                y=_create_bars_data_for_mixed_kde_plot(train_uniques_counts, np.max(test_density)),
                width=[bars_width] * train_uniques.size,
                marker=dict(color=colors[DEFAULT_DATASET_NAMES[0]]),
                name=dataset_names[0] + ' Dataset',
            ))
        else:
            traces.extend(_create_distribution_scatter_plot(xs, train_density, mean_train_column, median_train_column,
                                                            is_train=True, dataset_names=dataset_names))

        if test_uniques.size <= MAX_NUMERICAL_UNIQUES_FOR_SINGLE_DIST_BARS:
            traces.append(go.Bar(
                x=test_uniques,
                y=_create_bars_data_for_mixed_kde_plot(test_uniques_counts, np.max(train_density)),
                width=[bars_width] * test_uniques.size,
                marker=dict(
                    color=colors[DEFAULT_DATASET_NAMES[1]]
                ),
                name=dataset_names[1] + ' Dataset',
            ))
        else:
            traces.extend(_create_distribution_scatter_plot(xs, test_density, mean_test_column, median_test_column,
                                                            is_train=False, dataset_names=dataset_names))

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


def _create_distribution_scatter_plot(xs, ys, mean=None, median=None, is_train=True,
                                      dataset_names: Tuple[str] = DEFAULT_DATASET_NAMES) -> List[go.Scatter]:
    traces = []
    name = dataset_names[0] if is_train else dataset_names[1]
    train_or_test = DEFAULT_DATASET_NAMES[0] if is_train else DEFAULT_DATASET_NAMES[1]
    traces.append(go.Scatter(x=xs, y=ys, name=f'{name} Dataset', fill='tozeroy',
                             line=dict(color=colors[train_or_test], shape='linear')))
    if mean:
        y_mean_index = np.argmax(xs == mean)
        traces.append(go.Scatter(x=[mean, mean], y=[0, ys[y_mean_index]], name=f'{name} Mean',
                                 line=dict(color=colors[train_or_test], dash='dash'), mode='lines+markers'))
    if median:
        y_median_index = np.argmax(xs == median)
        traces.append(go.Scatter(x=[median, median], y=[0, ys[y_median_index]], name=f'{name} Median',
                                 line=dict(color=colors[train_or_test]), mode='lines'))
    return traces


def _create_distribution_bar_graphs(
        train_column: Union[np.ndarray, pd.Series],
        test_column: Union[np.ndarray, pd.Series],
        max_num_categories: int,
        show_categories_by: CategoriesSortingKind,
        dataset_names: Tuple[str] = DEFAULT_DATASET_NAMES
) -> Tuple[Any, Any]:
    """
    Create distribution bar graphs.

    Returns
    -------
    Tuple[Any, Any]:
        a tuple instance with figures traces, yaxis layout
    """
    expected, actual, categories_list = preprocess_2_cat_cols_to_same_bins(
        dist1=train_column,
        dist2=test_column,
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
    expected_percents, actual_percents, categories_list = zip(*distribution[:max_num_categories])

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
            marker=dict(color=colors[DEFAULT_DATASET_NAMES[0]]),
            name=dataset_names[0] + ' Dataset',
        ),
        go.Bar(
            x=cat_df.index,
            y=cat_df['Test dataset'],
            marker=dict(color=colors[DEFAULT_DATASET_NAMES[1]]),
            name=dataset_names[1] + ' Dataset',
        )
    ]

    yaxis_layout = dict(
        fixedrange=True,
        autorange=True,
        rangemode='normal',
        title='Frequency'
    )

    return traces, yaxis_layout


def get_text_outliers_graph(dist: Sequence, data: Sequence[str], lower_limit: float, upper_limit: float, dist_name: str,
                            is_categorical: bool):
    """Create a distribution / bar graph of the data and its outliers.

    Parameters
    ----------
    dist : Sequence
        The distribution of the data.
    data : Sequence[str]
        The data (used to give samples of it in hover).
    lower_limit : float
        The lower limit of the common part of the data (under it is an outlier).
    upper_limit : float
        The upper limit of the common part of the data (above it is an outlier).
    dist_name : str
        The name of the distribution (feature)
    is_categorical : bool
        Whether the data is categorical or not.
    """
    green = common_and_outlier_colors['common']
    red = common_and_outlier_colors['outliers']
    green_fill = common_and_outlier_colors['common_fill']
    red_fill = common_and_outlier_colors['outliers_fill']

    is_categorical = is_categorical or np.unique(dist).size < MAX_NUMERICAL_UNIQUE_FOR_BARS

    if is_categorical:
        dist_counts = pd.Series(dist).value_counts(normalize=True).to_dict()

        counts = list(dist_counts.values())
        categories_list = list(dist_counts.keys())

        outliers_first_index = counts.index(lower_limit)
        color_discrete_sequence = [green] * outliers_first_index + [red] * (len(counts) - outliers_first_index + 1)

        # fixes plotly widget bug with numpy values by converting them to native values
        # https://github.com/plotly/plotly.py/issues/3470
        cat_df = pd.DataFrame(
            {dist_name: counts},
            index=[un_numpy(cat) for cat in categories_list]
        )

        OUTLIER_LINE_INDEX = 'Outlier<br>Threshold'
        cat_df = pd.concat([cat_df.iloc[:outliers_first_index],
                            pd.DataFrame({dist_name: [None]}, index=[OUTLIER_LINE_INDEX]),
                            cat_df.iloc[outliers_first_index:]])

        tuples = list(zip(dist, data))

        tuples.sort(key=lambda x: x[0])
        samples_indices = np.searchsorted([x[0] for x in tuples], cat_df.index, side="left")
        samples = [tuples[i][1] for i in samples_indices]
        samples = [break_to_lines_and_trim(s) for s in samples]

        hover_data = np.array([samples, list(cat_df.index), list(cat_df[dist_name])]).T
        hover_template = f'<b>{dist_name}</b>: ' \
                         '%{customdata[1]}<br>' \
                         '<b>Frequency</b>: %{customdata[2]:.2%}<br>' \
                         '<b>Sample</b>:<br>"%{customdata[0]}"<br>'

        traces = [
            go.Bar(
                x=cat_df.index,
                y=cat_df[dist_name],
                marker=dict(color=color_discrete_sequence),
                name='Common',
                text=[f'{x:.2%}' for x in cat_df[dist_name]],
                customdata=hover_data,
                hovertemplate=hover_template

            ),
            go.Bar(  # Adding fake bar traces to show the outlier threshold line in the legend
                x=[None],
                y=[None],
                name='Outliers',
                marker=dict(color=red),
            ),
        ]

        yaxis_layout = dict(
            fixedrange=True,
            autorange=True,
            rangemode='normal',
            title='Frequency (Log Scale)',
            type='log'
        )
        xaxis_layout = dict(type='category')

    else:
        x_range = (
            min(dist.min(), dist.min()),
            max(dist.max(), dist.max())
        )

        # Heuristically take points on x-axis to show on the plot
        # The intuition is the graph will look "smooth" wherever we will zoom it
        # Also takes mean and median values in order to plot it later accurately
        xs = sorted(np.concatenate((
            np.linspace(x_range[0], x_range[1], 50),
            np.quantile(dist, q=np.arange(0.02, 1, 0.02))
        )))

        traces: List[go.BaseTraceType] = []

        # In order to plot the common and outliers parts of the graph in different colors, we need to separate them into
        # different traces. We do it by creating a mask for each part and then using it to filter the data.
        # However, for the graphs to start and end smoothly, we need to add a point in the beginning and end of the
        # common part. Those duplicate points will be set to start or end each trace in 0.
        all_arr = [1 if lower_limit <= x <= upper_limit else 0 for x in xs]
        common_beginning = all_arr.index(1)
        common_ending = len(all_arr) - 1 - all_arr[::-1].index(1)

        show_lower_outliers = common_beginning != 0
        show_upper_outliers = common_ending != len(xs) - 1
        total_len = len(xs) + show_lower_outliers + show_upper_outliers

        mask_common = np.zeros(total_len, dtype=bool)
        mask_outliers_lower = np.zeros(total_len, dtype=bool)
        mask_outliers_upper = np.zeros(total_len, dtype=bool)

        density = list(get_density(dist, xs))

        # If there are lower outliers, add a duplicate point to the beginning of the common part:
        if common_beginning != 0:
            xs.insert(common_beginning, xs[common_beginning])
            density.insert(common_beginning, density[common_beginning])
            mask_outliers_lower[:common_beginning + 1] = True
            common_ending += 1

        # If there are upper outliers, add a duplicate point to the end of the common part:
        if common_ending != len(xs) - 1:
            xs.insert(common_ending + 1, xs[common_ending])
            density.insert(common_ending + 1, density[common_ending])
            mask_outliers_upper[common_ending + 1:] = True

        mask_common[common_beginning + show_lower_outliers:common_ending + show_upper_outliers] = True

        density_common = np.array(density) * mask_common
        density_outliers_lower = np.array(density) * mask_outliers_lower
        density_outliers_upper = np.array(density) * mask_outliers_upper

        # Replace 0s with None so that they won't be plotted:
        density_common = [x or None for x in density_common]
        density_outliers_lower = [x or None for x in density_outliers_lower]
        density_outliers_upper = [x or None for x in density_outliers_upper]

        # Get samples and their quantiles for the hover data:
        tuples = list(zip(dist, data))
        tuples.sort(key=lambda x: x[0])
        samples_indices = np.searchsorted([x[0] for x in tuples], xs, side="left")
        samples = [tuples[i][1] for i in samples_indices]
        samples = [break_to_lines_and_trim(s) for s in samples]
        quantiles = [100 * i / len(dist) for i in samples_indices]
        hover_data = np.array([samples, xs, quantiles]).T
        hover_template = f'<b>{dist_name}</b>: ' \
                         '%{customdata[1]:.2f}<br>' \
                         '<b>Larger than</b> %{customdata[2]:.2f}% of samples<br>' \
                         '<b>Sample</b>:<br>"%{customdata[0]}"<br>'

        traces.append(go.Scatter(
            x=xs, y=density_common, name='Common', fill='tozeroy', fillcolor=green_fill,
            line=dict(color=green, shape='linear', width=5), customdata=hover_data, hovertemplate=hover_template
        ))
        traces.append(go.Scatter(
            x=xs, y=density_outliers_lower, name='Lower Outliers', fill='tozeroy', fillcolor=red_fill,
            line=dict(color=red, shape='linear', width=5), customdata=hover_data, hovertemplate=hover_template))

        traces.append(go.Scatter(
            x=xs, y=density_outliers_upper, name='Upper Outliers', fill='tozeroy', fillcolor=red_fill,
            line=dict(color=red, shape='linear', width=5), customdata=hover_data, hovertemplate=hover_template))

        xaxis_layout = dict(fixedrange=False,
                            title=dist_name)
        yaxis_layout = dict(title='Probability Density', fixedrange=True)

    fig = go.Figure(data=traces)
    fig.update_xaxes(xaxis_layout)
    fig.update_yaxes(yaxis_layout)

    if is_categorical: # Add vertical line to separate outliers from common values in bar charts:
        fig.add_vline(x=OUTLIER_LINE_INDEX, line_width=2, line_dash="dash", line_color="black")

    fig.update_layout(
        legend=dict(
            title='Legend',
            yanchor='top',
            y=0.6),
        height=400,
        title=dict(text=dist_name, x=0.5, xanchor='center'),
        bargroupgap=0,
        hovermode='closest')

    return fig
