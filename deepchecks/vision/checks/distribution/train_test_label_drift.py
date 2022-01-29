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
from copy import copy
from typing import Dict, Hashable, Callable, Tuple, List, Union

from plotly.subplots import make_subplots

from deepchecks import CheckResult, ConditionResult
from deepchecks.vision.base import TrainTestBaseCheck, Context
from deepchecks.utils.distribution.plot import drift_score_bar_traces
from deepchecks.utils.plot import colors
from deepchecks.vision.dataset import VisionDataset, TaskType
import numpy as np
from collections import Counter
import plotly.graph_objs as go


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

    def run_logic(self, context: Context) -> CheckResult:
        """
        Calculate drift for all columns.

        Args:
            train_dataset (VisionDataset): The training dataset object. Must contain a label.
            test_dataset (VisionDataset): The test dataset object. Must contain a label.

        Returns:
            CheckResult:
                value: drift score.
                display: label distribution graph, comparing the train and test distributions.
        """
        train_dataset = context.train
        test_dataset = context.test

        task_type = train_dataset.label_type
        displays = []

        if task_type == TaskType.CLASSIFICATION.value:

            train_label_distribution = train_dataset.get_samples_per_class()
            test_label_distribution = test_dataset.get_samples_per_class()

            drift_score, method, display = calc_drift_and_plot(
                train_distribution=train_label_distribution,
                test_distribution=test_label_distribution,
                plot_title='Class',
                column_type='categorical',
            )

            values_dict = {'Drift score': drift_score, 'Method': method}
            displays.append(display)

        elif task_type == TaskType.OBJECT_DETECTION.value:

            #TODO: This should be one process, that iterates over the dataset once, not every metric.
            """this means that histogram_in_batch and count_custom_transform_on_label should be the same func,
            and that it should receive multiple transforms and do them"""

            #TODO: Enable sampling of label distribution


            values_dict = {}

            # Drift on samples per class:
            title = 'Samples per class'
            train_label_distribution = train_dataset.get_samples_per_class()
            test_label_distribution = test_dataset.get_samples_per_class()

            drift_score, method, display = calc_drift_and_plot(
                train_distribution=train_label_distribution,
                test_distribution=test_label_distribution,
                plot_title=title,
                column_type='categorical',
            )

            values_dict[title] = {'Drift score': drift_score, 'Method': method}
            displays.append(display)

            # Drift on bbox areas:
            title = 'bbox area distribution'
            train_label_distribution = histogram_in_batch(dataset=train_dataset, label_transformer=get_bbox_area)
            test_label_distribution = histogram_in_batch(dataset=test_dataset, label_transformer=get_bbox_area)

            drift_score, method, display = calc_drift_and_plot(
                train_distribution=train_label_distribution,
                test_distribution=test_label_distribution,
                plot_title=title,
                column_type='numerical'
            )

            values_dict[title] = {'Drift score': drift_score, 'Method': method}
            displays.append(display)

            # Number of bboxes per image
            title = 'Number of bboxes per image'
            train_label_distribution = count_custom_transform_on_label(dataset=train_dataset,
                                                                       label_transformer=count_num_bboxes)
            test_label_distribution = count_custom_transform_on_label(dataset=test_dataset,
                                                                      label_transformer=count_num_bboxes)

            drift_score, method, display = calc_drift_and_plot(
                train_distribution=train_label_distribution,
                test_distribution=test_label_distribution,
                plot_title=title,
                column_type='categorical',
            )

            values_dict[title] = {'Drift score': drift_score, 'Method': method}
            displays.append(display)

        else:
            raise NotImplementedError('Currently not implemented') #TODO

        headnote = """<span>
            The Drift score is a measure for the difference between two distributions, in this check - the test
            and train distributions.<br> The check shows the drift score and distributions for the label.
        </span>"""

        displays = [headnote] + displays

        return CheckResult(value=values_dict, display=displays, header='Train Test Label Drift')

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
                return_str = f'Found label PSI above threshold: {drift_score:.2f}'
                return ConditionResult(False, return_str)
            elif method == "Earth Mover's Distance" and has_failed:
                return_str = f'Label\'s Earth Mover\'s Distance above threshold: {drift_score:.2f}'
                return ConditionResult(False, return_str)

            return ConditionResult(True)

        return self.add_condition(f'PSI <= {max_allowed_psi_score} and Earth Mover\'s Distance <= '
                                  f'{max_allowed_earth_movers_score} for label drift',
                                  condition)


def get_bbox_area(label):
    areas = (label.reshape((-1, 5))[:, 4] * label.reshape((-1, 5))[:, 3]).reshape(-1, 1).tolist()
    return areas


def count_num_bboxes(label):
    num_bboxes = label.shape[0]
    return num_bboxes


def count_custom_transform_on_label(dataset: VisionDataset, label_transformer: Callable = lambda x: x):
    counter = Counter()
    for i in range(len(dataset.get_data_loader())):
        list_of_arrays = next(iter(dataset.get_data_loader()))[1]
        calc_res = [label_transformer(arr) for arr in list_of_arrays]
        if len(calc_res) != 0 and isinstance(calc_res[0], list):
            calc_res = [x[0] for x in sum(calc_res, [])]
        counter.update(calc_res)
    return counter


def histogram_in_batch(dataset: VisionDataset, label_transformer: Callable = lambda x: x):
    label_min = np.inf
    label_max = -np.inf
    for i in range(len(dataset.get_data_loader())):
        list_of_arrays = next(iter(dataset.get_data_loader()))[1]
        calc_res = [label_transformer(arr) for arr in list_of_arrays]
        if len(calc_res) != 0 and isinstance(calc_res[0], list):
            calc_res = [x[0] for x in sum(calc_res, [])]
        label_min = min(calc_res + [label_min])
        label_max = max(calc_res + [label_max])

    hist, edges = np.histogram([], bins=100, range=(label_min, label_max))

    for i in range(len(dataset.get_data_loader())):
        list_of_arrays = next(iter(dataset.get_data_loader()))[1]
        calc_res = [label_transformer(arr) for arr in list_of_arrays]
        if len(calc_res) != 0 and isinstance(calc_res[0], list):
            calc_res = [x[0] for x in sum(calc_res, [])]
        new_hist, _ = np.histogram(calc_res, bins=100, range=(label_min, label_max))
        hist = new_hist + hist

    return dict(zip(edges, hist))


PSI_MIN_PERCENTAGE = 0.01

def psi(expected_percents: np.ndarray, actual_percents: np.ndarray):
    """
    Calculate the PSI (Population Stability Index).

    See https://www.lexjansen.com/wuss/2017/47_Final_Paper_PDF.pdf

    Args:
        expected_percents: array of percentages of each value in the expected distribution.
        actual_percents: array of percentages of each value in the actual distribution.

    Returns:
        psi: The PSI score

    """
    psi_value = 0
    for i in range(len(expected_percents)):
        # In order for the value not to diverge, we cap our min percentage value
        e_perc = max(expected_percents[i], PSI_MIN_PERCENTAGE)
        a_perc = max(actual_percents[i], PSI_MIN_PERCENTAGE)
        value = (e_perc - a_perc) * np.log(e_perc / a_perc)
        psi_value += value

    return psi_value


def earth_movers_distance_by_histogram(expected_percents: np.ndarray, actual_percents: np.ndarray):
    dirt = copy(actual_percents)
    delta = 1 / expected_percents.size
    emd = 0
    for i in range(dirt.shape[0]-1):
        dirt_to_pass = dirt[i] - expected_percents[i]
        dirt[i+1] += dirt_to_pass
        emd += abs(dirt_to_pass)*delta
    return emd


def calc_drift_and_plot(train_distribution: dict, test_distribution: dict, plot_title: Hashable,
                        column_type: str) -> Tuple[float, str, Callable]:
    """
    Calculate drift score per column.

    Args:
        train_distribution: column from train dataset
        test_distribution: same column from test dataset
        plot_title: title of plot
        column_type: type of column (either "numerical" or "categorical")
        max_num_categories: Max number of allowed categories. If there are more, they are binned into an "Other"
                            category.

    Returns:
        score: drift score of the difference between the two columns' distributions (Earth movers distance for
        numerical, PSI for categorical)
        display: graph comparing the two distributions (density for numerical, stack bar for categorical)
    """

    if column_type == 'numerical':
        scorer_name = "Earth Mover's Distance"

        expected_percents = np.array(list(train_distribution.values()))
        actual_percents = np.array(list(test_distribution.values()))

        score = earth_movers_distance_by_histogram(expected_percents=expected_percents, actual_percents=actual_percents)
        bar_traces, bar_x_axis, bar_y_axis = drift_score_bar_traces(score, bar_max=1)

        x_values = list(train_distribution.keys())

        dist_traces, dist_x_axis, dist_y_axis = feature_distribution_traces(expected_percents, actual_percents,
                                                                            x_values)

    elif column_type == 'categorical':
        scorer_name = 'PSI'

        categories_list = list(set(train_distribution.keys()).union(set(test_distribution.keys())))

        expected_percents = \
            np.array([train_distribution[k] for k in categories_list]) / np.sum(list(train_distribution.values()))
        actual_percents = \
            np.array([test_distribution[k] for k in categories_list]) / np.sum(list(test_distribution.values()))

        score = psi(expected_percents=expected_percents, actual_percents=actual_percents)

        bar_traces, bar_x_axis, bar_y_axis = drift_score_bar_traces(score, bar_max=1)

        dist_traces, dist_x_axis, dist_y_axis = feature_distribution_traces(expected_percents, actual_percents,
                                                                            categories_list, is_categorical=True)

    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.4, shared_yaxes=False, shared_xaxes=False,
                        row_heights=[0.1, 0.9],
                        subplot_titles=['Drift Score - ' + scorer_name, plot_title])

    fig.add_traces(bar_traces, rows=[1] * len(bar_traces), cols=[1] * len(bar_traces))
    fig.add_traces(dist_traces, rows=[2] * len(dist_traces), cols=[1] * len(dist_traces))

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
        height=400
    )


    fig.update_layout(shared_layout)

    return score, scorer_name, fig

def feature_distribution_traces(expected_percents: np.array,
                                actual_percents: np.array,
                                x_values: list,
                                is_categorical: bool = False) -> [List[Union[go.Bar, go.Scatter]], Dict, Dict]:
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

        train_bar = go.Bar(
            x=x_values,
            y=expected_percents,
            marker=dict(
                color=colors['Train'],
            ),
            name='Train Dataset',
        )

        test_bar = go.Bar(
            x=x_values,
            y=actual_percents,
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
        # pass
        x_range = (x_values[0], x_values[-1])
        xs = np.linspace(x_range[0], x_range[1], 40)

        traces = [go.Scatter(x=xs, y=expected_percents, fill='tozeroy', name='Train Dataset',
                             line_color=colors['Train']),
                  go.Scatter(x=xs, y=actual_percents, fill='tozeroy', name='Test Dataset',
                             line_color=colors['Test'])]

        xaxis_layout = dict(fixedrange=True,
                            range=x_range,
                            title='Distribution')
        yaxis_layout = dict(title='Probability Density')

    return traces, xaxis_layout, yaxis_layout
