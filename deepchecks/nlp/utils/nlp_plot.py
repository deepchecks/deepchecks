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
from typing import List, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

from deepchecks.nlp import TextData
from deepchecks.nlp.task_type import TaskType
from deepchecks.nlp.utils.text import break_to_lines_and_trim
from deepchecks.nlp.utils.text_properties import TEXT_PROPERTIES_DESCRIPTION
from deepchecks.nlp.utils.token_classification_utils import (annotated_token_classification_text,
                                                             count_token_classification_labels)
from deepchecks.utils.dataframes import un_numpy
from deepchecks.utils.distribution.plot import get_density
from deepchecks.utils.plot import DEFAULT_DATASET_NAMES, colors, common_and_outlier_colors
from deepchecks.utils.strings import get_docs_link

__all__ = ['get_text_outliers_graph',
           'two_datasets_scatter_plot']


def clean_x_axis_non_existent_values(x_axis, distribution):
    """Remove values from x_axis where the distribution has no values."""
    # Find the index of the first value in x_axis that is bigger than the value in distribution
    ixs = np.searchsorted(sorted(distribution), x_axis, side='left')
    # If 2 neighboring indexes are the same, it means that there are no values in the distribution for
    # the corresponding value in x_axis. We remove it.
    x_axis = [x_axis[i] for i in range(len(ixs)) if ixs[i] != ixs[i - 1]]
    return x_axis


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

        outlier_line_index = 'Outlier<br>Threshold'
        cat_df = pd.concat([cat_df.iloc[:outliers_first_index],
                            pd.DataFrame({dist_name: [None]}, index=[outlier_line_index]),
                            cat_df.iloc[outliers_first_index:]])

        # Get samples and their frequency for the hover data:
        tuples = list(zip(dist, data))
        tuples.sort(key=lambda x: x[0])
        samples_indices = np.searchsorted([x[0] for x in tuples], cat_df.index, side='left')
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
                text=[f'{x:.2%}' if x is not None else None for x in cat_df[dist_name]],
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
        dist = dist[~pd.isnull(dist)]

        x_range = (
            dist.min(), dist.max()
        )
        if all(int(x) == x for x in dist if x is not None):
            # If the distribution is discrete, we take all the values in it:
            xs = sorted(np.unique(dist))
            if len(xs) > 50:
                # If there are too many values, we take only 50, using a constant interval between them:
                xs = list(range(int(xs[0]), int(xs[-1]) + 1, int((xs[-1] - xs[0]) // 50)))
        else:
            # Heuristically take points on x-axis to show on the plot
            # The intuition is the graph will look "smooth" wherever we will zoom it
            # Also takes mean and median values in order to plot it later accurately
            xs = sorted(np.concatenate((
                np.linspace(x_range[0], x_range[1], 50),
                np.quantile(dist, q=np.arange(0.02, 1, 0.02))
            )))
            xs = clean_x_axis_non_existent_values(xs, dist)

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

        # Replace zeros (meaning, non-related values from the mask) with None so that they won't be plotted:
        density_common = [x or None for x in density_common]
        density_outliers_lower = [x or None for x in density_outliers_lower]
        density_outliers_upper = [x or None for x in density_outliers_upper]

        # Get samples and their quantiles for the hover data:
        tuples = list(zip(dist, data))
        tuples.sort(key=lambda x: x[0])
        samples_indices = np.searchsorted([x[0] for x in tuples], xs, side='left')
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
            line=dict(color=green, shape='linear', width=5), customdata=hover_data, hovertemplate=hover_template,
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

    if is_categorical:  # Add vertical line to separate outliers from common values in bar charts:
        fig.add_vline(x=outlier_line_index, line_width=2, line_dash='dash', line_color='black')

    if dist_name in TEXT_PROPERTIES_DESCRIPTION:
        dist_name = f'{dist_name}<sup><a href="{get_docs_link()}nlp/usage_guides/nlp_properties.html' \
                    '#deepchecks-built-in-properties">&#x24D8;</a></sup><br>' \
                    f'<sup>{TEXT_PROPERTIES_DESCRIPTION[dist_name]}</sup>'
    fig.update_layout(
        legend=dict(
            title='Legend',
            yanchor='top',
            y=0.6),
        height=400,
        title=dict(text=dist_name, x=0.5, xanchor='center'),
        bargroupgap=0,
        hovermode='closest',
        hoverdistance=-1)

    return fig


def two_datasets_scatter_plot(plot_title: str, plot_data: pd.DataFrame, train_dataset: TextData,
                              test_dataset: TextData, model_classes: list):
    """Plot a scatter plot of two datasets.

    Parameters
    ----------
    plot_title : str
        The title of the plot.
    plot_data : pd.DataFrame
        The data to plot (x and y axes).
    train_dataset : TextData
        The train dataset.
    test_dataset : TextData
        The test dataset.
    model_classes : list
        The names of the model classes (relevant only if the datasets are multi-label).
    """
    axes = plot_data.columns
    if train_dataset.name and test_dataset.name:
        dataset_names = (train_dataset.name, test_dataset.name)
    else:
        dataset_names = DEFAULT_DATASET_NAMES

    plot_data['Dataset'] = [dataset_names[0]] * len(train_dataset) + [dataset_names[1]] * len(test_dataset)

    if train_dataset.task_type == TaskType.TOKEN_CLASSIFICATION:
        plot_data['Sample'] = np.concatenate([train_dataset.tokenized_text, test_dataset.tokenized_text])

        if train_dataset.has_label():
            plot_data['Label'] = list(train_dataset.label_for_display(model_classes=model_classes)) + \
                                 list(test_dataset.label_for_display(model_classes=model_classes))
            plot_data['Sample'] = annotated_token_classification_text(plot_data['Sample'], plot_data['Label'])
            # Displayed labels are the counts of each label in the dataset:
            plot_data['Label'] = [break_to_lines_and_trim(str(count_token_classification_labels(x)))
                                  for x in plot_data['Label']]
        else:
            plot_data['Label'] = None
    else:
        if train_dataset.has_label():
            plot_data['Label'] = list(train_dataset.label_for_print(model_classes=model_classes)) + \
                                 list(test_dataset.label_for_print(model_classes=model_classes))
        else:
            plot_data['Label'] = None
        plot_data['Sample'] = np.concatenate([train_dataset.text, test_dataset.text])

    plot_data['Sample'] = plot_data['Sample'].apply(break_to_lines_and_trim)

    fig = px.scatter(plot_data, x=axes[0], y=axes[1], color='Dataset', color_discrete_map=colors,
                     hover_data=['Label', 'Sample'], hover_name='Dataset', title=plot_title, opacity=0.4)
    fig.update_traces(marker=dict(size=8, line=dict(width=1, color='DarkSlateGrey')), selector=dict(mode='markers'))
    return fig
