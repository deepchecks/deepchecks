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
"""Module of segment performance check."""
import typing as t
from numbers import Number

import numpy as np
import pandas as pd
import torch
from ignite.metrics import Metric
import plotly.express as px


from deepchecks.core import DatasetKind, CheckResult
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.utils.strings import format_number
from deepchecks.vision import SingleDatasetCheck, Context
from deepchecks.vision.utils import ImageFormatter
from deepchecks.vision.metrics_utils import get_scorers_list, metric_results_to_df


ImageProperty = t.Union[str, t.Callable[..., t.List[Number]]]

__all__ = ['SegmentPerformance']


class SegmentPerformance(SingleDatasetCheck):
    """
    """

    def __init__(
        self,
        image_properties: t.Optional[t.List[ImageProperty]] = None,
        alternative_metrics: t.Optional[t.Dict[str, Metric]] = None,
        number_of_bins: int = 5,
        number_of_samples_to_infer_bins: int = 1000
    ):
        super().__init__()

        if image_properties is None:
            self.image_properties = ImageFormatter.IMAGE_PROPERTIES
        else:
            if len(image_properties) == 0:
                raise DeepchecksValueError('image_properties list cannot be empty')

            properties_by_name = {p for p in image_properties if isinstance(p, str)}
            unknown_properties = properties_by_name.difference(ImageFormatter.IMAGE_PROPERTIES)

            if len(unknown_properties) > 0:
                raise DeepchecksValueError(
                    'received list of unknown image properties '
                    f'- {sorted(unknown_properties)}'
                )

            self.image_properties = image_properties

        self.alternative_metrics = alternative_metrics
        self.number_of_bins = number_of_bins
        self.number_of_samples_to_infer_bins = number_of_samples_to_infer_bins
        self._state = None

    def initialize_run(self, context: Context, dataset_kind: DatasetKind):
        """Initialize run before starting updating on batches"""
        # First we will aggregate samples up to defined amount (number_of_samples_to_infer_bins), when we reach
        # the amount we will define the bins and populate them
        self._state = {'samples_for_binning': [], 'bins': None}
        # Initialize image properties. Doing this not in the init because we use the dataset object.
        dataset = context.get_data_by_kind(dataset_kind)
        string_props = {p: getattr(dataset.image_formatter, p) for p in self.image_properties if isinstance(p, str)}
        func_props = {p.__name__: p for p in self.image_properties if callable(p)}
        self._state['properties_functions'] = {**string_props, **func_props}

    def update(self, context: Context, batch: t.Any, dataset_kind: DatasetKind):
        dataset = context.get_data_by_kind(dataset_kind)
        images = dataset.image_formatter(batch)
        predictions = context.infer(batch)
        labels = dataset.label_formatter(batch)

        samples_for_bin: t.List = self._state['samples_for_binning']
        bins = self._state['bins']
        properties_functions: t.Dict = self._state['properties_functions']

        # Initialize a list of all properties per image sample
        batch_properties = []
        for _ in range(len(images)):
            batch_properties.append({})
        for prop_name, func in properties_functions.items():
            for index, image_result in enumerate(func(images)):
                batch_properties[index][prop_name] = image_result

        batch_data = zip(labels, predictions, batch_properties)
        # If we already defined bins, add the current data to them
        if bins is not None:
            _divide_to_bins(bins, batch_data)
        else:
            # Add the current data to the samples list
            samples_for_bin.extend(batch_data)
            # Check if enough data to infer bins
            if len(samples_for_bin) < self.number_of_samples_to_infer_bins:
                # Create the bins and metrics, and divide all cached data into the bins
                self._state['bins'] = self._create_bins_and_metrics(samples_for_bin, dataset)
                # Remove the samples which are no longer needed (free the memory)
                del samples_for_bin

    def compute(self, context: Context, dataset_kind: DatasetKind) -> CheckResult:
        """Calculate segment performance based on image properties.

        Returns
        -------
        CheckResult
            value: dictionary containing performance for each property segments
            display: table of results
        """
        dataset = context.get_data_by_kind(dataset_kind)
        # In case there are fewer samples than 'number_of_samples_to_infer_bins' then bins were not calculated
        if self._state['bins'] is None:
            # Create the bins and metrics
            bins = self._create_bins_and_metrics(self._state['samples_for_binning'], dataset)
        else:
            bins = self._state['bins']

        # bins are in format:
        # {property_name: [{start: val, stop: val, count: x, metrics: {name: metric...}}, ...], ...}
        display_data = []
        metric_names = None

        for property_name, prop_bins in bins.items():
            for single_bin in prop_bins:
                # If we have a low number of unique values for a property, the first bin (-inf, x) might be empty so
                # check the count, and if empty filter out the bin
                if single_bin['count'] == 0:
                    continue

                bin_data = {
                    'Range': _range_string(single_bin['start'], single_bin['stop']),
                    'Number of samples': single_bin['count'],
                    'Property': f'Property: {property_name}'
                }
                # Update the metrics in the single bin from the metrics objects to metric mean results, in order to
                # return ths bins object as the check result value
                single_bin['metrics'] = _calculate_metrics(single_bin['metrics'], dataset)
                # For the plotly display need row per metric in the dataframe
                for metric, val in single_bin['metrics'].items():
                    display_data.append({'Metric': metric, 'Value': val, **bin_data})
                # Get the metric names to use in the plot creation
                if metric_names is None:
                    metric_names = list(single_bin['metrics'].keys())

        display_df = pd.DataFrame(display_data)

        fig = px.bar(
            display_df,
            x='Range',
            y='Value',
            color='Metric',
            barmode='group',
            facet_col='Property',
            facet_row='Metric',
            facet_col_spacing=0.05,
            facet_row_spacing=0.05,
            hover_data=['Number of samples']
        )

        (fig.update_xaxes(title=None, type='category', matches=None)
            .update_yaxes(title=None)
            .for_each_annotation(lambda a: a.update(text=a.text.split('=')[-1]))
            .for_each_yaxis(lambda yaxis: yaxis.update(showticklabels=True)))

        return CheckResult(value=bins, display=fig)

    def _create_bins_and_metrics(self, batch_data: t.List[t.Tuple], dataset):
        """Return dict of bins for each property in format
        {property_name: [{start: val, stop: val, count: x, metrics: {name: metric...}}, ...], ...}"""
        # For X bins we need to have (X - 1) quantile bounds (open bounds from left and right)
        quantiles = np.linspace(1 / self.number_of_bins, 1, self.number_of_bins - 1, endpoint=False)
        # Calculate for each property the quantile values
        batch_properties = [b[2] for b in batch_data]
        df = pd.DataFrame(batch_properties)
        bins = {}
        for prop in df.columns:
            # Filter nans
            # TODO: how to handle if only some of the values are nan?
            property_col = df[~df[prop].isnull()][prop]
            # If all values of the property are nan, then doesn't display it
            if len(property_col) == 0:
                continue
            # Get quantiles without duplicates
            quantile_values = list(set(df[prop].quantile(quantiles).tolist()))
            bins[prop] = [{'start': start, 'stop': stop, 'count': 0,
                          'metrics': get_scorers_list(dataset, self.alternative_metrics)}
                          for start, stop in _create_open_bins_ranges(quantile_values)]

        # Divide the data into the bins
        _divide_to_bins(bins, batch_data)
        return bins


def _divide_to_bins(bins, batch_data: t.Iterable[t.Tuple]):
    """Iterate the data and enter it into the appropriate bins."""
    for property_name, bins_values in bins.items():
        for label, prediction, properties in batch_data:
            _add_to_fitting_bin(bins_values, properties[property_name], label, prediction)


def _create_open_bins_ranges(quantiles):
    """Return quantiles with start and stop as list of tuples [(-Inf, x1),(x1,x2),(x2, Inf)]"""
    quantiles = sorted(quantiles)
    # if len(quantiles) == 1:
    #     return [(quantiles[0], np.Inf)]
    return zip(([-np.Inf] + quantiles), (quantiles + [np.Inf]))


def _add_to_fitting_bin(bins: t.List[t.Dict], property_value, label, prediction):
    """Find the fitting bin from the list of bins for a given value. Then increase the count and the prediction and
    label to the metrics objects."""
    if property_value is None:
        return
    for single_bin in bins:
        if single_bin['start'] <= property_value < single_bin['stop']:
            single_bin['count'] += 1
            for metric in single_bin['metrics'].values():
                # Since this is a single prediction and label need to wrap in tensor
                metric.update((torch.unsqueeze(prediction, 0), torch.unsqueeze(label, 0)))
            return


def _range_string(start, stop):
    start = '[' + format_number(start) if not np.isinf(start) else '(-inf'
    stop = format_number(stop) if not np.isinf(stop) else 'inf'
    return f'{start}, {stop})'


def _calculate_metrics(metrics, dataset):
    metrics_df = metric_results_to_df(
        {k: m.compute() for k, m in metrics.items()}, dataset
    )
    metrics_df = metrics_df[['Metric', 'Value']].groupby(['Metric']).median()
    return metrics_df.to_dict()['Value']


# def _make_plot(df, metrics):
#     fig = make_subplots(rows=1, cols=len(metrics))
#
#     for i, metric in enumerate(metrics):
#         bar = go.Bar(df, x=df['Range'], y=df[metric], )