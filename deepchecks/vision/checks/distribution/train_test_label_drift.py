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
    See https://en.wikipedia.org/wiki/Wasserstein_metric
    For categorical columns, we use the Population Stability Index (PSI).
    See https://www.lexjansen.com/wuss/2017/47_Final_Paper_PDF.pdf.


    Parameters
    ----------
    max_num_categories : int , default: 10
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
        """Calculate drift for all columns.

        Returns
        -------
        CheckResult
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


            # TODO: This should be one process, that iterates over the dataset once, not every metric.
            # this means that histogram_in_batch and count_custom_transform_on_label should be the same function,
            # and that it should receive multiple transforms and do them

            # TODO: Enable sampling of label distribution
            # TODO: Re-use max_num_categories

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

            continuous_label_transformers = [get_bbox_area]
            discrete_label_transformers = [count_num_bboxes]
            train_distributions, test_distributions = generate_label_histograms_by_batch(train_dataset=train_dataset, test_dataset=test_dataset,
                                                                                         continuous_label_transformers=continuous_label_transformers, discrete_label_transformers=discrete_label_transformers)

            for title, train_label_distribution, test_label_distribution, is_continuous in zip(['bbox area distribution', 'Number of bboxes per image'], train_distributions, test_distributions, [True, False]):

                drift_score, method, display = calc_drift_and_plot(
                    train_distribution=train_label_distribution,
                    test_distribution=test_label_distribution,
                    plot_title=title,
                    column_type='numerical' if is_continuous else 'categorical'
                )

                values_dict[title] = {'Drift score': drift_score, 'Method': method}
                displays.append(display)

        else:
            raise NotImplementedError('Currently not implemented')  # TODO

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

        Parameters
        ----------
        max_allowed_psi_score: float , default: 0.2
            the max threshold for the PSI score
        max_allowed_earth_movers_score: float ,  default: 0.1
            the max threshold for the Earth Mover's Distance score
        Returns
        -------
        ConditionResult
            False if any column has passed the max threshold, True otherwise
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


def generate_label_histograms_by_batch(train_dataset: VisionDataset, test_dataset: VisionDataset,
                                       continuous_label_transformers: List[Callable] = None,
                                       discrete_label_transformers: List[Callable] = None,
                                       num_bins: int = 100):

    if not continuous_label_transformers and not discrete_label_transformers:
        discrete_label_transformers = [lambda x: x]

    num_continuous_transformers = len(continuous_label_transformers)
    num_discrete_transformers = len(discrete_label_transformers)

    train_bounds = get_boundaries_by_batch(train_dataset, continuous_label_transformers)
    test_bounds = get_boundaries_by_batch(test_dataset, continuous_label_transformers)
    bounds = [(min(train_bounds[i]['min'], test_bounds[i]['min']),
               max(train_bounds[i]['max'], test_bounds[i]['max'])) for i in range(num_continuous_transformers)]

    hists_and_edges = [np.histogram([], bins=num_bins, range=(bound[0], bound[1])) for bound in bounds]
    train_hists = [x[0] for x in hists_and_edges]
    test_hists = copy(train_hists)
    edges = [x[1] for x in hists_and_edges]

    train_counters = [Counter()] * num_continuous_transformers
    test_counters = [Counter()] * num_continuous_transformers

    for batch in train_dataset.get_data_loader():
        train_hists = calculate_continuous_histograms_in_batch(batch, train_hists, continuous_label_transformers, bounds, num_bins)
        train_counters = calculate_discrete_histograms_in_batch(batch, train_counters, discrete_label_transformers)

    for batch in test_dataset.get_data_loader():
        test_hists = calculate_continuous_histograms_in_batch(batch, test_hists, continuous_label_transformers, bounds, num_bins)
        test_counters = calculate_discrete_histograms_in_batch(batch, test_counters, discrete_label_transformers)

    all_discrete_categories = [list(set(train_counter.keys()).union(set(test_counter.keys())))
                               for train_counter, test_counter in zip(train_counters, test_counters)]

    train_discrete_hists = [{k: train_counters[i][k] for k in all_discrete_categories[i]} for i in range(num_discrete_transformers)]
    test_discrete_hists = [{k: test_counters[i][k] for k in all_discrete_categories[i]} for i in range(num_discrete_transformers)]

    train_continuous_hists = [dict(zip(edges[i], train_hists[i])) for i in range(num_continuous_transformers)]
    test_continuous_hists = [dict(zip(edges[i], test_hists[i])) for i in range(num_continuous_transformers)]

    return train_continuous_hists + train_discrete_hists, test_continuous_hists + test_discrete_hists


def calculate_discrete_histograms_in_batch(batch, counters, discrete_label_transformers):
    for i in range(len(discrete_label_transformers)):
        calc_res = get_results_on_batch(batch, discrete_label_transformers[i])
        counters[i].update(calc_res)
    return counters


def calculate_continuous_histograms_in_batch(batch, hists, continuous_label_transformers, bounds, num_bins):
    for i in range(len(continuous_label_transformers)):
        calc_res = get_results_on_batch(batch, continuous_label_transformers[i])
        new_hist, _ = np.histogram(calc_res, bins=num_bins, range=(bounds[i][0], bounds[i][1]))
        hists[i] += new_hist
    return hists

def get_results_on_batch(batch, label_transformer):
        list_of_arrays = batch[1]
        calc_res = [label_transformer(arr) for arr in list_of_arrays]
        if len(calc_res) != 0 and isinstance(calc_res[0], list):
            calc_res = [x[0] for x in sum(calc_res, [])]
        return calc_res


def get_boundaries_by_batch(dataset: VisionDataset, label_transformers: List[Callable]) -> List[Dict[str, float]]:
    bounds = [{'min': np.inf, 'max': -np.inf}] * len(label_transformers)
    for batch in dataset.get_data_loader():
        for i in range(len(label_transformers)):
            calc_res = get_results_on_batch(batch, label_transformers[i])
            bounds[i]['min'] = min(calc_res + [bounds[i]['min']])
            bounds[i]['max'] = max(calc_res + [bounds[i]['max']])

    return bounds


PSI_MIN_PERCENTAGE = 0.01


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


def earth_movers_distance_by_histogram(expected_percents: np.ndarray, actual_percents: np.ndarray):
    """
    Calculate the Earth Movers Distance (Wasserstein distance) by histogram.

    See https://en.wikipedia.org/wiki/Wasserstein_metric

    Function is for numerical data only.

    Parameters
    ----------
    expected_percents : np.ndarray
        array of percentages of each value in the expected distribution.
    actual_percents : np.ndarray
        array of percentages of each value in the actual distribution.
    Returns
    -------
    Any
        the Wasserstein distance between the two distributions.

    """
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

    Parameters
    ----------
    train_distribution : dict
        histogram of train values
    test_distribution : dict
        matching histogram for test dataset values
    plot_title : Hashable
        title of plot
    column_type : str
        type of column (either "numerical" or "categorical")
    Returns
    -------
    Tuple[float, str, Callable]
        drift score of the difference between the two columns' distributions (Earth movers distance for
        numerical, PSI for categorical)
        graph comparing the two distributions (density for numerical, stack bar for categorical)
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
            np.array(list(train_distribution.keys())) / np.sum(list(train_distribution.values()))
        actual_percents = \
            np.array(list(test_distribution.keys())) / np.sum(list(test_distribution.values()))

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
                                is_categorical: bool = False) -> Tuple[List[Union[go.Bar, go.Scatter]], Dict, Dict]:
    """Create traces for comparison between train and test column.

    Parameters
    ----------
    expected_percents: np.array
        Expected distribution of values
    actual_percents: np.array
        Actual distribution of values
    x_values: list
        list of x-axis values of expected_percents and actual_percents
    is_categorical : bool , default: False
        State if column is categorical.
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
