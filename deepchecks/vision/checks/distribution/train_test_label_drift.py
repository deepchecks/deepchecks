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

from deepchecks.core import DatasetKind, CheckResult
from deepchecks.core.errors import DeepchecksValueError, DeepchecksNotSupportedError
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
def _get_bbox_area(label, _):
    """Return a list containing the area of bboxes per image in batch."""
    areas = (label.reshape((-1, 5))[:, 4] * label.reshape((-1, 5))[:, 3]).reshape(-1, 1).tolist()
    return areas


def _count_num_bboxes(label, _):
    """Return a list containing the number of bboxes per image in batch."""
    num_bboxes = label.shape[0]
    return num_bboxes


def _get_samples_per_class_classification(label, dataset):
    """Return a list containing the class per image in batch."""
    return dataset.label_id_to_name(label.tolist())


def _get_samples_per_class_object_detection(label, dataset):
    """Return a list containing the class per image in batch."""
    return [[dataset.label_id_to_name(arr.reshape((-1, 5))[:, 0])] for arr in label]


DEFAULT_CLASSIFICATION_LABEL_MEASUREMENTS = [
    {'name': 'Samples per class', 'method': _get_samples_per_class_classification, 'is_continuous': False}
]

DEFAULT_OBJECT_DETECTION_LABEL_MEASUREMENTS = [
    {'name': 'Samples per class', 'method': _get_samples_per_class_object_detection, 'is_continuous': False},
    {'name': 'Bounding box area (in pixels)', 'method': _get_bbox_area, 'is_continuous': True},
    {'name': 'Number of bounding boxes per image', 'method': _count_num_bboxes, 'is_continuous': True},
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
    min_sample_size: int, default: None
        number of minimum samples (not batches) to be accumulated in order to estimate the boundaries (min, max) of
        continuous histograms. As the check cannot load all label measurement results into memory, the check saves only
        the histogram of results - but prior to that, the check requires to know the estimated boundaries of train AND
        test datasets (they must share an x-axis).
    default_num_bins: int, default: 100
        number of bins to use for continuous distributions. This value is not used if the distribution has less unique
        values than default number of bins (and instead, number of unique values is used).
    """

    def __init__(
            self,
            alternative_label_measurements: List[Dict[str, Any]] = None,
            min_sample_size: int = None,
            default_num_bins: int = 100
    ):
        super().__init__()
        # validate alternative_label_measurements:
        if alternative_label_measurements is not None:
            self._validate_label_measurements(alternative_label_measurements)
        self.alternative_label_measurements = alternative_label_measurements
        self.min_sample_size = min_sample_size
        self.default_num_bins = default_num_bins

    def initialize_run(self, context: Context):
        """Initialize run.

        Function initializes the following private variables:

        Label measurements:
        _label_measurements: all label measurements to be calculated in run
        _continuous_label_measurements: all continuous label measurements
        _discrete_label_measurements: all discrete label measurements

        Value counts of measures, to be updated per batch:
        _train_hists, _test_hists: histograms for continuous measurements for train and test respectively.
            Initialized as list of empty histograms (np.array) that update in the "update" method per batch.
        _train_counters, _test_counters: counters for discrete measurements for train and test respectively.
            Initialized as list of empty counters (collections.Counter) that update in the "update" method per batch.

        Parameters for continuous measurements histogram calculation:
        _bounds_list: List[Tuple]. Each tuple represents histogram bounds (min, max)
        _num_bins_list: List[int]. List of number of bins for each histogram.
        _edges: List[np.array]. List of x-axis values for each histogram.
        """
        train_dataset = context.train
        test_dataset = context.test

        task_type = train_dataset.task_type

        if self.alternative_label_measurements is not None:
            self._label_measurements = self.alternative_label_measurements
        elif task_type == TaskType.CLASSIFICATION:
            self._label_measurements = DEFAULT_CLASSIFICATION_LABEL_MEASUREMENTS
        elif task_type == TaskType.OBJECT_DETECTION:
            self._label_measurements = DEFAULT_OBJECT_DETECTION_LABEL_MEASUREMENTS
        else:
            raise NotImplementedError('TrainTestLabelDrift must receive either alternative_label_measurements or run '
                                      'on Classification or Object Detection class')

        # Separate to discrete and continuous transformers:
        self._continuous_label_measurements = [d['method'] for d in self._label_measurements if
                                               d['is_continuous'] is True]
        self._discrete_label_measurements = [d['method'] for d in self._label_measurements if
                                             d['is_continuous'] is False]

        num_continuous_transformers = len(self._continuous_label_measurements)
        num_discrete_transformers = len(self._discrete_label_measurements)

        # For continuous transformers, calculate bounds:
        train_bounds = get_boundaries_by_batch(train_dataset, self._continuous_label_measurements, self.min_sample_size)
        test_bounds = get_boundaries_by_batch(test_dataset, self._continuous_label_measurements, self.min_sample_size)
        bounds = [(min(train_bounds[i]['min'], test_bounds[i]['min']),
                   max(train_bounds[i]['max'], test_bounds[i]['max'])) for i in range(num_continuous_transformers)]

        self._bounds_list, self._num_bins_list = adjust_bounds_and_bins(bounds, self.default_num_bins)

        hists_and_edges = [np.histogram([], bins=num_bins, range=bound) for bound, num_bins in
                           zip(self._bounds_list, self._num_bins_list)]
        self._train_hists = [x[0] for x in hists_and_edges]
        self._test_hists = [copy(hist) for hist in self._train_hists]
        self._edges = [x[1] for x in hists_and_edges]

        self._train_counters = [Counter() for i in range(num_discrete_transformers)]
        self._test_counters = [Counter() for i in range(num_discrete_transformers)]

    def update(self, context: Context, batch: Any, dataset_kind):
        """Perform update on batch for train or test counters and histograms."""
        # For all transformers, calculate histograms by batch:
        if dataset_kind == DatasetKind.TRAIN:
            train_dataset = context.train
            self._train_hists = calculate_continuous_histograms_in_batch(batch, self._train_hists,
                                                                         self._continuous_label_measurements,
                                                                         self._bounds_list, self._num_bins_list,
                                                                         train_dataset)
            self._train_counters = calculate_discrete_histograms_in_batch(batch, self._train_counters,
                                                                          self._discrete_label_measurements,
                                                                          train_dataset)

        elif dataset_kind == DatasetKind.TEST:
            test_dataset = context.test
            self._test_hists = calculate_continuous_histograms_in_batch(batch, self._test_hists,
                                                                        self._continuous_label_measurements,
                                                                        self._bounds_list, self._num_bins_list,
                                                                        test_dataset)
            self._test_counters = calculate_discrete_histograms_in_batch(batch, self._test_counters,
                                                                         self._discrete_label_measurements,
                                                                         test_dataset)
        else:
            raise DeepchecksNotSupportedError(f'Unsupported dataset kind {dataset_kind}')

    def compute(self, context: Context) -> CheckResult:
        """Calculate drift on label measurements histograms that were collected during update() calls.

        Returns
        -------
        CheckResult
            value: drift score.
            display: label distribution graph, comparing the train and test distributions.
        """
        # Match discrete histograms to share x axis:
        all_discrete_categories = [list(set(train_counter.keys()).union(set(test_counter.keys())))
                                   for train_counter, test_counter in zip(self._train_counters, self._test_counters)]

        train_discrete_hists = iter([{k: self._train_counters[i][k]
                                      for k in all_discrete_categories[i]}
                                    for i in range(len(self._discrete_label_measurements))])
        test_discrete_hists = iter([{k: self._test_counters[i][k]
                                     for k in all_discrete_categories[i]}
                                    for i in range(len(self._discrete_label_measurements))])

        # Transform continuous histograms into dict:
        train_continuous_hists = iter(
            [dict(zip(self._edges[i], self._train_hists[i])) for i in range(len(self._continuous_label_measurements))])
        test_continuous_hists = iter(
            [dict(zip(self._edges[i], self._test_hists[i])) for i in range(len(self._continuous_label_measurements))])

        # # Return output in original order:
        train_distributions = [
            next(train_continuous_hists) if d['is_continuous'] is True else next(train_discrete_hists) for d in
            self._label_measurements]
        test_distributions = [next(test_continuous_hists) if d['is_continuous'] is True else next(test_discrete_hists)
                              for d in self._label_measurements]

        values_dict = {}
        displays = []

        figs_configs = zip(self._label_measurements, train_distributions, test_distributions)
        for d, train_label_distribution, test_label_distribution in figs_configs:
            drift_score, method, display = calc_drift_and_plot(
                train_distribution=train_label_distribution,
                test_distribution=test_label_distribution,
                plot_title=d['name'],
                column_type='numerical' if d['is_continuous'] else 'categorical'
            )

            values_dict[d['name']] = {'Drift score': drift_score, 'Method': method}
            displays.append(display)
        label_properties = [x['name'] for x in self._label_measurements]
        headnote = '<span>' \
                   'The Drift score is a measure for the difference between two distributions. ' \
                   'In this check, drift is measured ' \
                   f'for the distribution of the following label properties: {label_properties}.' \
                   '</span>'

        displays = [headnote] + displays

        return CheckResult(value=values_dict, display=displays, header='Train Test Label Drift')

    @staticmethod
    def _validate_label_measurements(label_measurements):
        """Validate structure of label measurements."""
        expected_keys = ['name', 'method', 'is_continuous']
        if not isinstance(label_measurements, list):
            raise DeepchecksValueError(
                f'Expected label measurements to be a list, instead got {label_measurements.__class__.__name__}')
        for label_measurement in label_measurements:
            if not isinstance(label_measurement, dict) or any(
                    key not in label_measurement.keys() for key in expected_keys):
                raise DeepchecksValueError(f'Label measurement must be of type dict, and include keys {expected_keys}')


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


def calculate_discrete_histograms_in_batch(batch, counters, discrete_label_measurements, dataset):
    """Calculate discrete histograms by batch."""
    for i in range(len(discrete_label_measurements)):
        calc_res = get_results_on_batch(batch, discrete_label_measurements[i], dataset)
        counters[i].update(calc_res)
    return counters


def calculate_continuous_histograms_in_batch(batch, hists, continuous_label_measurements, bounds, bins,
                                             dataset):
    """Calculate continuous histograms by batch."""
    for i in range(len(continuous_label_measurements)):
        calc_res = get_results_on_batch(batch, continuous_label_measurements[i], dataset)
        new_hist, _ = np.histogram(calc_res, bins=bins[i], range=(bounds[i][0], bounds[i][1]))
        hists[i] += new_hist
    return hists


def get_results_on_batch(batch, label_measurement, dataset: VisionData):
    """Calculate transformer result on batch of labels."""
    calc_res = [label_measurement(arr, dataset) for arr in dataset.label_formatter(batch)]
    if len(calc_res) != 0 and isinstance(calc_res[0], list):
        calc_res = [x[0] for x in sum(calc_res, [])]
    return calc_res


def get_boundaries_by_batch(dataset: VisionData, label_measurements: List[Callable], min_sample_size: int) \
        -> List[Dict[str, float]]:
    """Get min and max on dataset for each label transformer."""
    bounds = [{'min': np.inf, 'max': -np.inf} for _ in range(len(label_measurements))]
    num_samples = 0
    for batch in dataset.get_data_loader():
        for i in range(len(label_measurements)):
            calc_res = get_results_on_batch(batch, label_measurements[i], dataset)
            bounds[i]['min'] = min(calc_res + [bounds[i]['min']])
            bounds[i]['max'] = max(calc_res + [bounds[i]['max']])

        num_samples += len(batch[0])
        if min_sample_size is not None and num_samples >= min_sample_size:
            return bounds

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

    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.2, shared_yaxes=False, shared_xaxes=False,
                        row_heights=[0.1, 0.9],
                        subplot_titles=['Drift Score - ' + scorer_name, 'Distribution Plot'])

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
        height=400,
        title=plot_title
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

        max_y = max(max(expected_percents), max(actual_percents))
        y_lim = 1 if max_y > 0.5 else max_y * 1.1

        xaxis_layout = dict(type='category',
                            title='Value')
        yaxis_layout = dict(fixedrange=True,
                            range=(0, y_lim),
                            title='Percentage')

    else:
        x_range = (x_values[0], x_values[-1])

        traces = [go.Scatter(x=x_values, y=expected_percents, fill='tozeroy', name='Train Dataset',
                             line_color=colors['Train']),
                  go.Scatter(x=x_values, y=actual_percents, fill='tozeroy', name='Test Dataset',
                             line_color=colors['Test'])]

        xaxis_layout = dict(fixedrange=True,
                            range=x_range,
                            title='Value')
        yaxis_layout = dict(title='Probability Density')

    return traces, xaxis_layout, yaxis_layout
