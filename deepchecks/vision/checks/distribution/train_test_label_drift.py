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
"""Module contains Train Test label Drift check."""
from copy import copy
from typing import Dict, Hashable, Callable, Tuple, List, Union, Any

from plotly.subplots import make_subplots

from deepchecks import CheckResult
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.vision.base import Context, TrainTestCheck
from deepchecks.utils.distribution.plot import drift_score_bar_traces
from deepchecks.utils.plot import colors
from deepchecks.vision.dataset import VisionData, TaskType
import numpy as np
from collections import Counter
import plotly.graph_objs as go

__all__ = ['TrainTestLabelDrift']


# TODO: Add label sampling when available

# Functions temporarily here, will be changed when Label and Prediction classes exist:
def get_bbox_area(label):
    """Return a list containing the area of bboxes per image in batch."""
    areas = (label.reshape((-1, 5))[:, 4] * label.reshape((-1, 5))[:, 3]).reshape(-1, 1).tolist()
    return areas


def count_num_bboxes(label):
    """Return a list containing the number of bboxes per image in batch."""
    num_bboxes = label.shape[0]
    return num_bboxes


def get_samples_per_class_classification(label):
    """Return a list containing the class per image in batch."""
    return label.tolist()


def get_samples_per_class_object_detection(label):
    """Return a list containing the class per image in batch."""
    return [arr.reshape((-1, 5))[:, 0].tolist() for arr in label]


DEFAULT_CLASSIFICATION_LABEL_MEASUREMENTS = [
    {'name': 'Samples per class', 'method': get_samples_per_class_classification, 'is_continuous': False}
]

DEFAULT_OBJECT_DETECTION_LABEL_MEASUREMENTS = [
    {'name': 'Bounding box area distribution', 'method': get_bbox_area, 'is_continuous': True},
    {'name': 'Samples per class', 'method': get_samples_per_class_object_detection, 'is_continuous': False},
    {'name': 'Number of bounding boxes per image', 'method': count_num_bboxes, 'is_continuous': True},
]


class TrainTestLabelDrift(TrainTestCheck):
    """
    Calculate label drift between train dataset and test dataset, using statistical measures.

    Check calculates a drift score for the label in test dataset, by comparing its distribution to the train
    dataset. As the label may be complex, we run different measurements on the label and check their distribution.

    A measurement on a label is any function that returns a single value or n-dimensional array of values. each value
    represents a measurement on the label, such as number of objects in image or tilt of each bounding box in image.

    There are default measurements per task:
    For classification:
    - distribution of classes

    For object detection:
    - distribution of classes
    - distribution of bounding box areas
    - distribution of number of bounding boxes per image

    For numerical distributions, we use the Earth Movers Distance.
    See https://en.wikipedia.org/wiki/Wasserstein_metric
    For categorical distributions, we use the Population Stability Index (PSI).
    See https://www.lexjansen.com/wuss/2017/47_Final_Paper_PDF.pdf.


    Parameters
    ----------
    alternative_label_measurements : List[Dict[str, Any]], default: 10
        List of measurements. Replaces the default deepchecks measurements.
        Each measurement is dictionary with keys 'name' (str), 'method' (Callable) and is_continuous (bool),
        representing attributes of said method.
    num_bins: int, default: 100
            number of bins to use for continuous distributions
    """

    def __init__(
            self,
            alternative_label_measurements: List[Dict[str, Any]] = None,
            num_bins: int = 100
    ):
        super().__init__()
        # validate alternative_label_measurements:
        if alternative_label_measurements is not None:
            self._validate_label_measurements(alternative_label_measurements)
        self.alternative_label_measurements = alternative_label_measurements
        self.num_bins = num_bins

    def run_logic(self, context: Context) -> CheckResult:
        """Calculate drift for all columns.

        Returns
        -------
        CheckResult
            value: drift score.
            display: label distribution graph, comparing the train and test distributions.
        """
        np.random.seed(42)
        train_dataset = context.train
        test_dataset = context.test

        task_type = train_dataset.task_type
        displays = []

        if self.alternative_label_measurements is not None:
            label_measurements_list = self.alternative_label_measurements
        elif task_type == TaskType.CLASSIFICATION:
            label_measurements_list = DEFAULT_CLASSIFICATION_LABEL_MEASUREMENTS
        elif task_type == TaskType.OBJECT_DETECTION:
            label_measurements_list = DEFAULT_OBJECT_DETECTION_LABEL_MEASUREMENTS
        else:
            raise NotImplementedError('TrainTestLabelDrift must receive either alternative_label_measurements or run '
                                      'on Classification or Object Detection class')

        train_distributions, test_distributions = \
            generate_label_histograms_by_batch(train_dataset=train_dataset, test_dataset=test_dataset,
                                               label_measurements=label_measurements_list, num_bins=self.num_bins)

        figs_configs = zip(label_measurements_list, train_distributions, test_distributions)
        values_dict = {}

        for d, train_label_distribution, test_label_distribution in figs_configs:
            drift_score, method, display = calc_drift_and_plot(
                train_distribution=train_label_distribution,
                test_distribution=test_label_distribution,
                plot_title=d['name'],
                column_type='numerical' if d['is_continuous'] else 'categorical'
            )

            values_dict[d['name']] = {'Drift score': drift_score, 'Method': method}
            displays.append(display)

        headnote = """<span>
            The Drift score is a measure for the difference between two distributions, in this check - the test
            and train distributions of different measurement(s) on the label.
        </span>"""

        displays = [headnote] + displays

        return CheckResult(value=values_dict, display=displays, header='Train Test Label Drift')

    def _validate_label_measurements(self, label_measurements):
        """Validate structure of label measurements."""
        expected_keys = ['name', 'method', 'is_continuous']
        if not isinstance(label_measurements, list):
            raise DeepchecksValueError(
                f'Expected label measurements to be a list, instead got {label_measurements.__class__.__name__}')
        for label_measurement in label_measurements:
            if not isinstance(label_measurement, dict) or any(
                    key not in label_measurement.keys() for key in expected_keys):
                raise DeepchecksValueError(f'Label measurement must be of type dict, and include keys {expected_keys}')


def generate_label_histograms_by_batch(train_dataset: VisionData, test_dataset: VisionData,
                                       label_measurements: List[Dict[str, Any]] = None,
                                       num_bins: int = 100) -> Tuple[List[Dict[Any, float]], List[Dict[Any, float]]]:
    """
    Generate label histograms by received label transformers.

    This function calculates all label transformers per batch.
    For continuous transformers, the function has to run twice, once to get boundaries of histogram and second to
    calculate histograms. For discrete transformers, function runs only once.

    Parameters
    ----------
    train_dataset: VisionData
        dataset representing train data
    test_dataset: VisionData
        dataset representing test data
    label_measurements: List[Dict[str, Any]]
        list of measurements. Each measurement is dictionary with keys 'name' (str), 'method' (Callable) and
        is_continuous (bool), representing attributes of said method.
    num_bins: int, default 100
        number of bins to use for continuous distributions

    Returns
    -------
    Tuple[List[Dict[Any, float], List[Dict[Any, float]]]
        two lists of train and test histograms (each histogram is a dictionary, where key is returned metric or binned
        metric result, and value is the number of occurrences)

    """
    # Separate to discrete and continuous transformers:
    if not label_measurements:
        continuous_label_measurements = []
        discrete_label_measurements = [lambda x: x]
    else:
        continuous_label_measurements = [d['method'] for d in label_measurements if d['is_continuous'] is True]
        discrete_label_measurements = [d['method'] for d in label_measurements if d['is_continuous'] is False]

    num_continuous_transformers = len(continuous_label_measurements)
    num_discrete_transformers = len(discrete_label_measurements)

    # For continuous transformers, calculate bounds:
    train_bounds = get_boundaries_by_batch(train_dataset, continuous_label_measurements)
    test_bounds = get_boundaries_by_batch(test_dataset, continuous_label_measurements)
    bounds = [(min(train_bounds[i]['min'], test_bounds[i]['min']),
               max(train_bounds[i]['max'], test_bounds[i]['max'])) for i in range(num_continuous_transformers)]

    bounds, bins = adjust_bounds_and_bins(bounds, num_bins)

    hists_and_edges = [np.histogram([], bins=num_bins, range=bound) for bound, num_bins in
                       zip(bounds, bins)]
    train_hists = [x[0] for x in hists_and_edges]
    test_hists = [copy(hist) for hist in train_hists]
    edges = [x[1] for x in hists_and_edges]

    train_counters = [Counter() for i in range(num_discrete_transformers)]
    test_counters = [Counter() for i in range(num_discrete_transformers)]

    # For all transformers, calculate histograms by batch:
    for batch in train_dataset.get_data_loader():
        train_hists = calculate_continuous_histograms_in_batch(batch, train_hists, continuous_label_measurements,
                                                               bounds, bins, train_dataset.label_transformer)
        train_counters = calculate_discrete_histograms_in_batch(batch, train_counters, discrete_label_measurements,
                                                                train_dataset.label_transformer)

    for batch in test_dataset.get_data_loader():
        test_hists = calculate_continuous_histograms_in_batch(batch, test_hists, continuous_label_measurements, bounds,
                                                              bins, test_dataset.label_transformer)
        test_counters = calculate_discrete_histograms_in_batch(batch, test_counters, discrete_label_measurements,
                                                               test_dataset.label_transformer)

    # Match discrete histograms to share x axis:
    all_discrete_categories = [list(set(train_counter.keys()).union(set(test_counter.keys())))
                               for train_counter, test_counter in zip(train_counters, test_counters)]

    train_discrete_hists = iter([{k: train_counters[i][k] for k in all_discrete_categories[i]} for i in
                                 range(num_discrete_transformers)])
    test_discrete_hists = iter([{k: test_counters[i][k] for k in all_discrete_categories[i]} for i in
                                range(num_discrete_transformers)])

    # Transform continuous histograms into dict:
    train_continuous_hists = iter([dict(zip(edges[i], train_hists[i])) for i in range(num_continuous_transformers)])
    test_continuous_hists = iter([dict(zip(edges[i], test_hists[i])) for i in range(num_continuous_transformers)])

    # # Return output in original order:
    train_hists = [next(train_continuous_hists) if d['is_continuous'] is True else next(train_discrete_hists) for d in
                   label_measurements]
    test_hists = [next(test_continuous_hists) if d['is_continuous'] is True else next(test_discrete_hists) for d in
                  label_measurements]

    return train_hists, test_hists


def adjust_bounds_and_bins(bounds: List[Tuple[float, float]], default_num_bins: int) \
        -> Tuple[List[Tuple[float, float]], List[int]]:
    """Return adjusted bounds and bins for better presentation in graphs."""
    bins = [default_num_bins] * len(bounds)
    for i in range(len(bounds)):
        bmin, bmax = bounds[i]
        # If bounds are integers, we assume data is discrete integers and we'd like the binned data to reflect that:
        if np.floor(bmin) == bmin and np.floor(bmax) == bmax:
            if bmax - bmin < default_num_bins:
                bins[i] = int(bmax - bmin)  # Adjusted number of bins is the assumed number of unique values
            else:
                # If bounds are wider than the default_num_bins, adjust the upper bounds so that the bounds' range
                # is a round multiplication of default_num_ bins.
                # e.g. if bounds are (0, 197) and default_num_bins = 100, then change bounds to (0, 200).
                res = default_num_bins - (bmax - bmin) % default_num_bins
                bounds[i] = (bmin, bmax + res)
    return bounds, bins


def calculate_discrete_histograms_in_batch(batch, counters, discrete_label_measurements, label_transformer):
    """Calculate discrete histograms by batch."""
    for i in range(len(discrete_label_measurements)):
        calc_res = get_results_on_batch(batch, discrete_label_measurements[i], label_transformer)
        counters[i].update(calc_res)
    return counters


def calculate_continuous_histograms_in_batch(batch, hists, continuous_label_measurements, bounds, bins,
                                             label_transformer):
    """Calculate continuous histograms by batch."""
    for i in range(len(continuous_label_measurements)):
        calc_res = get_results_on_batch(batch, continuous_label_measurements[i], label_transformer)
        new_hist, _ = np.histogram(calc_res, bins=bins[i], range=(bounds[i][0], bounds[i][1]))
        hists[i] += new_hist
    return hists


def get_results_on_batch(batch, label_measurement, label_transformer):
    """Calculate transformer result on batch of labels."""
    list_of_arrays = batch[1]
    calc_res = [label_measurement(arr) for arr in label_transformer(list_of_arrays)]
    if len(calc_res) != 0 and isinstance(calc_res[0], list):
        calc_res = [x[0] for x in sum(calc_res, [])]
    return calc_res


def get_boundaries_by_batch(dataset: VisionData, label_measurements: List[Callable]) -> List[Dict[str, float]]:
    """Get min and max on dataset for each label transformer."""
    bounds = [{'min': np.inf, 'max': -np.inf} for i in range(len(label_measurements))]
    for batch in dataset.get_data_loader():
        for i in range(len(label_measurements)):
            calc_res = get_results_on_batch(batch, label_measurements[i], dataset.label_transformer)
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
    dirt = copy(actual_percents) / sum(actual_percents)
    dirt_to_match = copy(expected_percents) / sum(expected_percents)
    delta = 1 / expected_percents.size

    emd = 0
    for i in range(dirt.shape[0] - 1):
        dirt_to_pass = dirt[i] - dirt_to_match[i]
        dirt[i + 1] += dirt_to_pass
        emd += abs(dirt_to_pass) * delta
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
            np.array(list(train_distribution.values())) / np.sum(list(train_distribution.values()))
        actual_percents = \
            np.array(list(test_distribution.values())) / np.sum(list(test_distribution.values()))

        score = psi(expected_percents=expected_percents, actual_percents=actual_percents)

        bar_traces, bar_x_axis, bar_y_axis = drift_score_bar_traces(score, bar_max=1)

        dist_traces, dist_x_axis, dist_y_axis = feature_distribution_traces(expected_percents, actual_percents,
                                                                            categories_list, is_categorical=True)

    else:
        raise DeepchecksValueError(
            f'column_type must be one of ["numerical", "categorical"], instead got {column_type}')

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

        xaxis_layout = dict(type='category',
                            title='Values')
        yaxis_layout = dict(fixedrange=True,
                            range=(0, 1),
                            title='Percentage')

    else:
        x_range = (x_values[0], x_values[-1])

        traces = [go.Scatter(x=x_values, y=expected_percents, fill='tozeroy', name='Train Dataset',
                             line_color=colors['Train']),
                  go.Scatter(x=x_values, y=actual_percents, fill='tozeroy', name='Test Dataset',
                             line_color=colors['Test'])]

        xaxis_layout = dict(fixedrange=True,
                            range=x_range,
                            title='Distribution')
        yaxis_layout = dict(title='Probability Density')

    return traces, xaxis_layout, yaxis_layout
