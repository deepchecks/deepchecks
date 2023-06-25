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
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from deepchecks.nlp.task_type import TaskType, TTextLabel
from deepchecks.nlp.utils.text import break_to_lines_and_trim
from deepchecks.nlp.utils.text_properties import TEXT_PROPERTIES_DESCRIPTION
from deepchecks.utils.dataframes import un_numpy
from deepchecks.utils.distribution.plot import get_density
from deepchecks.utils.plot import feature_distribution_colors
from deepchecks.utils.strings import format_percent, get_docs_link

__all__ = ['text_data_describe_plot']


def _calculate_annoation_ratio(label, n_samples, is_mutli_label, task_type):

    if label is None:
        return format_percent(0)
    if is_mutli_label or task_type == TaskType.TOKEN_CLASSIFICATION:
        annotated_count = _calculate_number_of_annotated_samples(label=label,
                                                                 is_multi_label=is_mutli_label,
                                                                 task_type=task_type)
        return format_percent(annotated_count / n_samples)
    else:
        return format_percent(pd.notna(label).sum() / n_samples)


def _get_table_row_data(n_samples, annotation_ratio, categorical_metadata, numerical_metadata,
                        categorical_properties, numerical_properties, max_values_to_show: int = 5):

    info_cell = [n_samples, annotation_ratio]

    if categorical_metadata is None or len(categorical_metadata) == 0:
        info_cell.append('No categorical metadata')
    else:
        info_cell.append(', '.join(categorical_metadata) if len(categorical_metadata) <= max_values_to_show
                         else f'{len(categorical_metadata)} metadata columns')

    if numerical_metadata is None or len(numerical_metadata) == 0:
        info_cell.append('No numerical metadata')
    else:
        info_cell.append(', '.join(numerical_metadata) if len(numerical_metadata) <= max_values_to_show
                         else f'{len(numerical_metadata)} metadata columns')

    if categorical_properties is None or len(categorical_properties) == 0:
        info_cell.append('No categorical properties')
    else:
        info_cell.append(', '.join(categorical_properties) if len(categorical_properties) <= max_values_to_show
                         else f'{len(categorical_properties)} properties')

    if numerical_properties is None or len(numerical_properties) == 0:
        info_cell.append('No numerical properties')
    else:
        info_cell.append(', '.join(numerical_properties) if len(numerical_properties) <= max_values_to_show
                         else f'{len(numerical_properties)} properties')

    return info_cell


def _generate_table_trace(n_samples, annotation_ratio, categorical_metadata, numerical_metadata,
                          categorical_properties, numerical_properties):
    data_cell = ['<b>Number of samples</b>', '<b>Annotation ratio</b>', '<b>Metadata categorical columns</b>',
                 '<b>Metadata numerical columns</b>', '<b>Categorical properties</b>', '<b>Numerical properties</b>']

    info_cell = _get_table_row_data(n_samples=n_samples, annotation_ratio=annotation_ratio,
                                    categorical_metadata=categorical_metadata, numerical_metadata=numerical_metadata,
                                    categorical_properties=categorical_properties,
                                    numerical_properties=numerical_properties, max_values_to_show=7)

    trace = go.Table(header={'fill': {'color': 'white'}},
                     cells={'values': [data_cell, info_cell], 'align': ['left'], 'font_size': 12,
                            'height': 30})
    return trace


def _generate_categorical_distribution_plot(data, property_name):

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


def _calculate_number_of_annotated_samples(label, is_multi_label, task_type):

    if is_multi_label or task_type == TaskType.TOKEN_CLASSIFICATION:
        annotated_count = 0
        for label_data in label:
            annotated_count = annotated_count + 1 if len(label_data) > 0 and pd.isna(label_data).sum() == 0 \
                              else annotated_count
        return annotated_count
    else:
        return pd.notna(label).sum()


def _generate_numeric_distribution_plot(data, x_value, y_value, property_name):

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


def text_data_describe_plot(n_samples: int, max_num_labels_to_show: int,
                            is_multi_label: bool, task_type: str,
                            properties: pd.DataFrame,
                            categorical_metadata: Optional[List[str]] = None,
                            numerical_metadata: Optional[List[str]] = None,
                            categorical_properties: Optional[List[str]] = None,
                            numerical_properties: Optional[List[str]] = None,
                            model_classes: Optional[List[str]] = None,
                            label: Optional[TTextLabel] = None):
    """Return a plotly figure instance.

    Parameters
    ----------
    properties: pd.DataFrame
        The DataFrame consisting of the text properties data. If no prooperties are there, you can pass an
        empty DataFrame as well.
    n_samples: int
        The total number of samples present in the TextData object.
    max_num_labels_to_show : int
        The threshold to display the maximum number of labels on the label distribution pie chart and display
        rest of the labels under "Others" category.
    is_multi_label: bool
        A boolean where True denotes that the TextData contains multi labeled data otherwise false.
    task_type: str
        The task type for the text data. Can be either 'text_classification' or 'token_classification'.
    categorical_metadata: Optional[List[str]], default: None
        The names of the categorical metadata columns.
    numerical_metadata: Optional[List[str]], default: None
        The names of the numerical metadata columns.
    categorical_properties: Optional[List[str]], default: None
        The names of the categorical properties columns.
    numerical_properties: Optional[List[str]], default: None
        The names of the numerical text properties columns.
    label: Optional[TTextLabel], default: None
        The label for the text data. Can be either a text_classification label or a token_classification label.
        If None, the label distribution graph is not generated.

        - text_classification label - For text classification the accepted label format differs between multilabel and
          single label cases. For single label data, the label should be passed as a sequence of labels, with one entry
          per sample that can be either a string or an integer. For multilabel data, the label should be passed as a
          sequence of sequences, with the sequence for each sample being a binary vector, representing the presence of
          the i-th label in that sample.
        - token_classification label - For token classification the accepted label format is the IOB format or similar
          to it. The Label must be a sequence of sequences of strings or integers, with each sequence corresponding to
          a sample in the tokenized text, and exactly the length of the corresponding tokenized text.
    model_classes: Optional[List[str]], default: None
        List of classes names to use for multi-label display. Only used if the dataset is multi-label.

    Returns
    -------
    Plotly Figure instance.
    """
    specs = [[{'type': 'pie'}, {'type': 'table'}] if label is not None else [{'type': 'table', 'colspan': 2}, None]] + \
        [[{'type': 'xy', 'colspan': 2}, None] for _ in range(len(properties.columns))]

    subplot_titles = []
    if label is not None:
        annotated_samples = _calculate_number_of_annotated_samples(label, is_multi_label, task_type)
        subplot_titles.append(f'Label Distribution<br><sup>Out of {annotated_samples} annotated samples</sup><br><br>')

    subplot_titles.append('')  # Empty title for table figure
    if not properties.empty:
        for prop_name in properties:
            if prop_name in TEXT_PROPERTIES_DESCRIPTION:
                subplot_titles.append(f'{prop_name} Property Distribution<sup><a href="{get_docs_link()}nlp/'
                                      'usage_guides/nlp_properties.html#deepchecks-built-in-properties">&#x24D8;</a>'
                                      f'</sup><br><sup>{TEXT_PROPERTIES_DESCRIPTION[prop_name]}</sup>')

    fig = make_subplots(rows=len(properties.columns) + 1, cols=2, specs=specs, subplot_titles=subplot_titles,
                        row_heights=[1.5] + [1.0] * len(properties.columns))

    # Create label distribution if label is provided
    if label is not None:
        if is_multi_label:
            df_label = pd.DataFrame(label).fillna(0)
            if model_classes is not None:
                hashmap = {}
                for val in label:
                    model_array = np.array([model_classes[i] for i, val in enumerate(val) if val == 1])
                    for class_name in model_array:
                        hashmap[class_name] = hashmap[class_name] + 1 if class_name in hashmap else 1
                label_counts = pd.Series(list(hashmap.values()), index=list(hashmap))
            else:
                label_counts = pd.Series(np.sum(df_label.to_numpy(), axis=0))
        elif task_type == TaskType.TOKEN_CLASSIFICATION:
            hashmap = {}
            for val in label:
                flattened_array = pd.Series(np.array(val).flatten()).fillna('NaN').to_numpy()
                unique_values, counts = np.unique(flattened_array, return_counts=True)
                for label_value, count in zip(unique_values, counts):
                    if label_value != 'NaN':
                        hashmap[label_value] = hashmap[label_value] + count if label_value in hashmap else count
            label_counts = pd.Series(list(hashmap.values()), index=list(hashmap))
        else:
            label_counts = pd.Series(label).value_counts()

        label_counts.sort_values(ascending=False, inplace=True)
        labels_to_display = label_counts[:max_num_labels_to_show]
        labels_to_display.index = [break_to_lines_and_trim(str(label)) for label in list(labels_to_display.index)]
        count_other_labels = label_counts[max_num_labels_to_show + 1:].sum()
        labels_to_display['Others'] = count_other_labels

        # Pie chart for label distribution
        fig.add_trace(go.Pie(labels=list(labels_to_display.index), values=list(labels_to_display),
                             textposition='inside', showlegend=False, textinfo='label+percent',
                             hovertemplate='%{label}: %{value} samples<extra></extra>'), row=1, col=1)

    # Table figure for displaying some statistics
    annotation_ratio = _calculate_annoation_ratio(label, n_samples, is_multi_label, task_type)
    table_trace = _generate_table_trace(n_samples, annotation_ratio, categorical_metadata, numerical_metadata,
                                        categorical_properties, numerical_properties)
    fig.add_trace(table_trace, row=1, col=2 if label is not None else 1)

    # Looping over all the properties to generate respective property distribution graphs
    curr_row = 2  # Since row 1 is occupied with Pie and Table
    for property_name in properties.columns:

        if property_name in categorical_properties:
            # Creating bar plots for categorical properties
            trace, xaxis_layout, yaxis_layout = _generate_categorical_distribution_plot(
                                                    properties[property_name], property_name
                                                )
            fig.add_trace(trace, row=curr_row, col=1)
            fig.update_xaxes(xaxis_layout, row=curr_row, col=1)
            fig.update_yaxes(yaxis_layout, row=curr_row, col=1)
        else:
            # Creating scatter plots for numerical properties
            y_value, xs = _get_distribution_values(properties[property_name])
            trace, shapes, annotations, xaxis_layout, yaxis_layout = _generate_numeric_distribution_plot(
                                                                        properties[property_name],
                                                                        xs, y_value, property_name
                                                                    )
            fig.add_trace(trace, row=curr_row, col=1)

            for shape, annotation in zip(shapes, annotations):
                fig.add_shape(shape, row=curr_row, col=1)
                fig.add_annotation(annotation, row=curr_row, col=1)

            fig.update_yaxes(yaxis_layout, row=curr_row, col=1)
            fig.update_xaxes(xaxis_layout, row=curr_row, col=1)

        curr_row += 1

    fig.update_layout(height=450*(len(properties.columns) + 1))
    return fig
