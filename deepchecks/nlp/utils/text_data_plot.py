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
"""A module containing utils for displaying information on TextData object."""
from typing import List

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from deepchecks.nlp.utils.text_properties import TEXT_PROPERTIES_DESCRIPTION
from deepchecks.utils.dataframes import un_numpy
from deepchecks.utils.distribution.plot import get_density
from deepchecks.utils.plot import feature_distribution_colors
from deepchecks.utils.strings import format_percent, get_docs_link

__all__ = ['text_data_describe_plot']


def _generate_table_trace(n_samples, label, categorical_metadata, numerical_metadata,
                          categorical_properties, numerical_properties):
    annotation_ratio = format_percent(pd.notna(label).sum() / n_samples)
    data_cell = ['<b>Number of samples</b>', '<b>Annotation ratio</b>', '<b>Metadata categorical columns</b>',
                 '<b>Metadata numerical columns</b>', '<b>Categorical properties</b>', '<b>Numerical properties</b>']
    info_cell = [n_samples, annotation_ratio, ', '.join(categorical_metadata), ', '.join(numerical_metadata),
                 ', '.join(categorical_properties), ', '.join(numerical_properties)]
    trace = go.Table(header={'fill': {'color': 'white'}},
                     cells={'values': [data_cell, info_cell], 'align': ['left'], 'font_size': 12,
                            'height': 30})
    return trace


def _generate_bar_distribution_trace_text_properties(data, property_name):

    dist_counts = data.value_counts(normalize=True).to_dict()
    counts = list(dist_counts.values())
    categories_list = list(dist_counts.keys())
    cat_df = pd.DataFrame({property_name: counts}, index=[un_numpy(cat) for cat in categories_list])
    trace = go.Bar(x=cat_df.index, y=cat_df[property_name], showlegend=False,
                   marker={'color': feature_distribution_colors['feature']},
                   hovertemplate='<b>Value:</b> %{x}<br><b>Frequency:</b> %{y}<extra></extra>')
    yaxis_layout = dict(type='log', title='Frequency (Log Scale)')
    xaxis_layout = dict(title=property_name)
    return trace, xaxis_layout, yaxis_layout


def _get_distribution_values(data):
    mean = data.mean()
    median = data.median()
    x_range = (data.min(), data.max())
    if all(int(x) == x for x in data if x is not None):
        # If the distribution is discrete, we take all the values in it:
        xs = sorted(np.unique(data))
        if len(xs) > 50:
            # If there are too many values, we take only 50, using a constant interval between them:
            xs = list(range(int(xs[0]), int(xs[-1]) + 1, int((xs[-1] - xs[0]) // 50)))
    else:
        xs = sorted(np.concatenate((np.linspace(x_range[0], x_range[1], 50),
                                    np.quantile(data, q=np.arange(0.02, 1, 0.02)),
                                    [mean, median]
                                    )))
        ixs = np.searchsorted(sorted(data), xs, side='left')
        xs = [xs[i] for i in range(len(ixs)) if ixs[i] != ixs[i - 1]]
    y_value = get_density(data, xs)
    return y_value, xs


def _generate_scatter_distribution_trace(data, x_value, y_value, property_name):

    mean = data.mean()
    percentile_90 = data.quantile(0.9)
    percentile_10 = data.quantile(0.1)
    median = data.median()

    trace = go.Scatter(x=x_value, y=y_value, fill='tozeroy', showlegend=False,
                       hovertemplate=f'<b>{property_name}:</b> ''%{x}<br><b>Density:</b> %{y}<extra></extra>',
                       line={'color': feature_distribution_colors['feature'],
                             'shape': 'linear', 'width': 5})
    shapes = []
    annotations = []

    shapes.append(dict(type='line', x0=mean, y0=0, x1=mean, y1=max(y_value),
                       line={'color': feature_distribution_colors['measure'], 'dash': 'dash', 'width': 3}))
    mean_xpos = mean + max(x_value) * 0.02 if median < mean else mean - max(x_value) * 0.02
    annotations.append(dict(x=mean_xpos, y=max(y_value)/2, text='<b>Mean</b>', showarrow=False,
                            textangle=-90, font={'size': 12}))

    shapes.append(dict(type='line', x0=median, y0=0, x1=median, y1=max(y_value),
                       line={'color': feature_distribution_colors['measure'], 'dash': 'dot', 'width': 3}))
    median_xpos = median - max(x_value) * 0.02 if median < mean else median + max(x_value) * 0.02
    annotations.append(dict(x=median_xpos, y=max(y_value)/2, text='<b>Median</b>', showarrow=False,
                            textangle=-90, font={'size': 12}))

    shapes.append(dict(type='line', x0=percentile_10, y0=0, x1=percentile_10, y1=max(y_value),
                       line={'color': feature_distribution_colors['measure'], 'dash': 'dashdot', 'width': 3}))
    annotations.append(dict(x=percentile_10 - max(x_value)*0.02, y=max(y_value)/2, textangle=-90,
                            text='<b>10<sup>th</sup> Percentile</b>', showarrow=False, font={'size': 12}))

    shapes.append(dict(type='line', x0=percentile_90, y0=0, x1=percentile_90, y1=max(y_value),
                       line={'color': feature_distribution_colors['measure'], 'dash': 'dashdot', 'width': 3}))
    annotations.append(dict(x=percentile_90 + max(x_value)*0.02, y=max(y_value)/2, textangle=-90,
                            text='<b>90<sup>th</sup> Percentile</b>', showarrow=False, font={'size': 12}))

    xaxis_layout = dict(title=property_name)
    yaxis_layout = dict(title='Density')

    return trace, shapes, annotations, xaxis_layout, yaxis_layout


def text_data_describe_plot(all_properties_data, n_samples, label, categorical_metadata, numerical_metadata,
                            categorical_properties, numerical_properties, properties: List[str]):
    """Return a plotly figure instance.

    Parameters
    ----------
    all_properties_data:
    n_samples:
    label:
    categorical_metadata:
    numerical_metadata:
    categorical_properties:
    numerical_properties:
    properties : List[str]
        List of property names to consider for generating property distribution graphs.

    Returns
    -------
    Plotly Figure instance.
    """
    specs = [[{'type': 'pie'}, {'type': 'table'}]] + \
        [[{'type': 'xy', 'colspan': 2}, None] for _ in range(len(properties))]

    subplot_titles = [f'Label Distribution<br><sup>Out of {pd.notna(label).sum()} annotated samples</sup><br><br>',
                      '']
    for prop in properties:
        if prop in TEXT_PROPERTIES_DESCRIPTION:
            subplot_titles.append(f'{prop} Property Distribution<sup><a href="{get_docs_link()}nlp/usage_guides/'
                                  'nlp_properties.html#deepchecks-built-in-properties">&#x24D8;</a></sup><br>'
                                  f'<sup>{TEXT_PROPERTIES_DESCRIPTION[prop]}</sup>')

    fig = make_subplots(rows=len(properties) + 1, cols=2, specs=specs, subplot_titles=subplot_titles)
    label_counts = pd.Series(label).value_counts()

    # Pie chart for label distribution
    fig.add_trace(go.Pie(labels=list(label_counts.index), values=list(label_counts), textposition='inside',
                         hovertemplate='%{label}: %{value} samples<extra></extra>', textinfo='label+percent',
                         showlegend=False), row=1, col=1)

    # Table figure for displaying some statistics
    table_trace = _generate_table_trace(n_samples, label, categorical_metadata, numerical_metadata,
                                        categorical_properties, numerical_properties)
    fig.add_trace(table_trace, row=1, col=2)

    # Looping over all the properties to generate respective property distribution graphs
    curr_row = 2  # Since row 1 is occupied with Pie and Table
    for property_name in properties:

        if property_name in categorical_properties:
            # Creating bar plots for categorical properties
            trace, xaxis_layout, yaxis_layout = _generate_bar_distribution_trace_text_properties(
                                                    all_properties_data[property_name], property_name
                                                )
            fig.add_trace(trace, row=curr_row, col=1)
            fig.update_xaxes(xaxis_layout, row=curr_row, col=1)
            fig.update_yaxes(yaxis_layout, row=curr_row, col=1)
        else:
            # Creating scatter plots for numerical properties
            y_value, xs = _get_distribution_values(all_properties_data[property_name])
            trace, shapes, annotations, xaxis_layout, yaxis_layout = _generate_scatter_distribution_trace(
                                                                        all_properties_data[property_name],
                                                                        xs, y_value, property_name
                                                                    )
            fig.add_trace(trace, row=curr_row, col=1)

            for shape, annotation in zip(shapes, annotations):
                fig.add_shape(shape, row=curr_row, col=1)
                fig.add_annotation(annotation, row=curr_row, col=1)

            fig.update_yaxes(yaxis_layout, row=curr_row, col=1)
            fig.update_xaxes(xaxis_layout, row=curr_row, col=1)

        curr_row += 1

    fig.update_layout(height=400*(len(properties) + 1))
    return fig
