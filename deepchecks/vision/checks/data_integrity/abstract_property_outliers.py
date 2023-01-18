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
"""Module contains AbstractPropertyOutliers check."""
import string
import typing as t
import warnings
from abc import abstractmethod
from collections import defaultdict
from numbers import Number
from secrets import choice

import numpy as np
import pandas as pd

from deepchecks.core import CheckResult, DatasetKind
from deepchecks.core.errors import DeepchecksProcessError, NotEnoughSamplesError
from deepchecks.utils.outliers import iqr_outliers_range
from deepchecks.utils.strings import format_number
from deepchecks.vision.base_checks import SingleDatasetCheck
from deepchecks.vision.context import Context
from deepchecks.vision.utils.image_functions import draw_image
from deepchecks.vision.utils.vision_properties import PropertiesInputType
from deepchecks.vision.vision_data import TaskType, VisionData
from deepchecks.vision.vision_data.batch_wrapper import BatchWrapper

__all__ = ['AbstractPropertyOutliers']


class AbstractPropertyOutliers(SingleDatasetCheck):
    """Find outliers samples with respect to the given properties.

    The check computes several properties and then computes the number of outliers for each property.
    The check uses `IQR <https://en.wikipedia.org/wiki/Interquartile_range#Outliers>`_ to detect outliers out of the
    single dimension properties.

    Parameters
    ----------
    properties : List[Dict[str, Any]], default: None
        List of properties. Replaces the default deepchecks properties.
        Each property is a dictionary with keys ``'name'`` (str), ``method`` (Callable) and ``'output_type'`` (str),
        representing attributes of said method. 'output_type' must be one of:

        - ``'numeric'`` - for continuous ordinal outputs.
        - ``'categorical'`` - for discrete, non-ordinal outputs. These can still be numbers,
          but these numbers do not have inherent value.
        - ``'class_id'`` - for properties that return the class_id. This is used because these
          properties are later matched with the ``VisionData.label_map``, if one was given.

        For more on image / label properties, see the guide about :ref:`vision_properties_guide`.
    property_input_type: PropertiesInputType, default: PropertiesInputType.IMAGES
        The type of input to the properties, required for caching the results after first calculation.
    n_show_top : int , default: 3
        number of outliers to show from each direction (upper limit and bottom limit)
    iqr_percentiles: Tuple[int, int], default: (25, 75)
        Two percentiles which define the IQR range
    iqr_scale: float, default: 1.5
        The scale to multiply the IQR range for the outliers detection
    draw_label_on_image: bool, default: True
        Whether to draw the label on the image displayed or not.
    """

    def __init__(self,
                 properties_list: t.List[t.Dict[str, t.Any]] = None,
                 property_input_type: PropertiesInputType = PropertiesInputType.IMAGES,
                 n_show_top: int = 3,
                 iqr_percentiles: t.Tuple[int, int] = (25, 75),
                 iqr_scale: float = 1.5,
                 draw_label_on_image: bool = True,
                 n_samples: t.Optional[int] = 10000,
                 **kwargs):
        super().__init__(**kwargs)
        self.properties_list = properties_list
        self.property_input_type = property_input_type
        self.iqr_percentiles = iqr_percentiles
        self.iqr_scale = iqr_scale
        self.n_show_top = n_show_top
        self._draw_label_on_image = draw_label_on_image
        self._properties_results = None
        self.n_samples = n_samples

    def initialize_run(self, context: Context, dataset_kind: DatasetKind):
        """Initialize the properties state."""
        data = context.get_data_by_kind(dataset_kind)
        self._properties_results = defaultdict(list)
        # Dict of properties names to a dict of containing keys of property values, images
        self._lowest_property_value_images = defaultdict(list)
        self._highest_property_value_images = defaultdict(list)
        self._images_uuid = []

        self.properties_list = self.properties_list if self.properties_list else self.get_default_properties(data)
        if self.properties_list is not None and any(p['output_type'] == 'class_id' for p in self.properties_list):
            warnings.warn('Properties that have class_id as output_type will be skipped.')
            self.properties_list = [p for p in self.properties_list if p['output_type'] != 'class_id']

    def update(self, context: Context, batch: BatchWrapper, dataset_kind: DatasetKind):
        """Aggregate image properties from batch."""
        batch_properties = batch.vision_properties(self.properties_list, self.property_input_type)
        data = context.get_data_by_kind(dataset_kind)
        for prop_name, property_values in batch_properties.items():
            _ensure_property_shape(property_values, len(batch), prop_name)
            # If the property or label is single value per image, wrap them in order to work on a fixed structure
            if batch.numpy_labels is not None and data.task_type == TaskType.CLASSIFICATION:
                labels = [[label_per_image] for label_per_image in batch.numpy_labels]
            else:
                labels = batch.numpy_labels

            self._images_uuid += batch.numpy_image_identifiers
            self._cache_property_values_and_images(batch.numpy_images, labels, list(property_values), prop_name)

    def compute(self, context: Context, dataset_kind: DatasetKind) -> CheckResult:
        """Compute final result."""
        data = context.get_data_by_kind(dataset_kind)
        check_result = {}
        self._images_uuid = np.asarray(self._images_uuid)

        for name, values in self._properties_results.items():
            values_lengths_cumsum = np.cumsum(np.array([len(v) for v in values]))
            values_arr = np.hstack(values).astype(float)

            try:
                lower_limit, upper_limit = iqr_outliers_range(values_arr, self.iqr_percentiles, self.iqr_scale)
            except NotEnoughSamplesError:
                check_result[name] = 'Not enough non-null samples to calculate outliers.'
                continue

            outlier_values_idx = np.argwhere((values_arr < lower_limit) | (values_arr > upper_limit)).squeeze(axis=1)
            outlier_img_idx = np.unique([_sample_index_from_flatten_index(values_lengths_cumsum, outlier_index)
                                         for outlier_index in outlier_values_idx])
            outlier_img_identifiers = self._images_uuid[outlier_img_idx] if len(outlier_img_idx) > 0 else []
            check_result[name] = {
                'outliers_identifiers': outlier_img_identifiers,
                'lower_limit': max(lower_limit, min(values_arr)),
                'upper_limit': min(upper_limit, max(values_arr)),
            }

        # Create display
        if context.with_display:
            display = []
            no_outliers = pd.Series([], dtype='str')
            for property_name, info in check_result.items():
                # If info is string it means there was error
                if isinstance(info, str):
                    no_outliers = pd.concat([no_outliers, pd.Series(property_name, index=[info])])
                elif len(info['outliers_identifiers']) == 0:
                    no_outliers = pd.concat([no_outliers, pd.Series(property_name, index=['No outliers found.'])])
                else:
                    # Create id of alphabetic characters
                    images_and_values = self._get_property_outlier_images(property_name,
                                                                          info['lower_limit'], info['upper_limit'],
                                                                          data)
                    sid = ''.join([choice(string.ascii_uppercase) for _ in range(6)])
                    values_combine = ''.join([f'<div class="{sid}-item">{format_number(x[0])}</div>'
                                              for x in images_and_values])
                    images_combine = ''.join([f'<div class="{sid}-item">{x[1]}</div>'
                                              for x in images_and_values])

                    html = HTML_TEMPLATE.format(
                        prop_name=property_name,
                        values=values_combine,
                        images=images_combine,
                        count=len(info['outliers_identifiers']),
                        n_of_images=len(images_and_values),
                        lower_limit=format_number(info['lower_limit']),
                        upper_limit=format_number(info['upper_limit']),
                        id=sid
                    )

                    display.append(html)
            display = [''.join(display)]

            if not no_outliers.empty:
                grouped = no_outliers.groupby(level=0).unique().str.join(', ')
                grouped_df = pd.DataFrame(grouped, columns=['Properties'])
                grouped_df['More Info'] = grouped_df.index
                grouped_df = grouped_df[['More Info', 'Properties']]
                display.append('<h5><b>Properties With No Outliers Found</h5></b>')
                display.append(grouped_df.style.hide(axis='index') if hasattr(grouped_df.style, 'hide') else
                               grouped_df.style.hide_index())

        else:
            display = None

        return CheckResult(check_result, display=display)

    def _get_property_outlier_images(self, prop_name: str, lower_limit: float, upper_limit: float,
                                     vision_data) -> t.List[t.Tuple[float, str]]:
        """Get outlier images and their values for provided property."""
        result = []
        for idx, value in enumerate(self._lowest_property_value_images[prop_name]['property_values']):
            value = value[0] if isinstance(value, t.Sequence) else value  # for property per bbox, value is a list
            if value < lower_limit:
                image_thumbnail = draw_image(image=self._lowest_property_value_images[prop_name]['images'][idx],
                                             label=self._lowest_property_value_images[prop_name]['labels'][idx],
                                             task_type=vision_data.task_type, draw_label=self._draw_label_on_image,
                                             label_map=vision_data.label_map)
                result.append((value, image_thumbnail))
        for idx, value in enumerate(self._highest_property_value_images[prop_name]['property_values']):
            value = value[0] if isinstance(value, t.Sequence) else value  # for property per bbox, value is a list
            if value > upper_limit:
                image_thumbnail = draw_image(
                    image=self._highest_property_value_images[prop_name]['images'][idx],
                    label=self._highest_property_value_images[prop_name]['labels'][idx],
                    task_type=vision_data.task_type, draw_label=self._draw_label_on_image,
                    label_map=vision_data.label_map)
                result.append((value, image_thumbnail))
        return result

    @abstractmethod
    def get_default_properties(self, data: VisionData):
        """Return default properties to run in the check."""
        pass

    def _cache_property_values_and_images(self, images: t.List, labels: t.List, property_values: t.List,
                                          property_name: str):
        """Update the _lowest_property_value_images, _lowest_property_value_images dicts based on new batch."""
        is_property_per_label = isinstance(property_values[0], (np.ndarray, t.Sequence))
        # Update full property values cache for outlier calculation
        if is_property_per_label:
            self._properties_results[property_name].extend(property_values)
        else:
            self._properties_results[property_name].extend([[x] for x in property_values])
        # In case there are no images or no labels put none instead and do not display images / labels
        images = [None] * len(property_values) if images is None else images
        if labels is None:
            labels = [[None]] * len(property_values) if is_property_per_label else [None] * len(property_values)

        # adds the current lowest and highest property value images/labels/values to the batch before sorting
        if property_name in self._lowest_property_value_images:
            for stored_values_dict in (self._lowest_property_value_images[property_name],
                                       self._highest_property_value_images[property_name]):
                labels = stored_values_dict['labels'] + labels
                images = stored_values_dict['images'] + images
                property_values = stored_values_dict['property_values'] + property_values

        if is_property_per_label:  # if property is per label flatten the list of lists to find lowest and highest
            values_lengths_cumsum = np.cumsum(np.array([len(v) for v in property_values]))
            property_values = np.hstack(property_values).astype(float)
            labels = np.asarray([item for sublist in labels for item in sublist], dtype='object')

        # calculate lowest and highest property values
        not_null_indices = [idx for idx, value in enumerate(property_values) if value is not None]
        if len(not_null_indices) <= self.n_show_top:
            lowest_values_idx = not_null_indices
            highest_values_idx = not_null_indices
        else:
            lowest_values_idx = np.argpartition([np.inf if v is None else v for v in property_values],
                                                self.n_show_top)[:self.n_show_top]
            highest_values_idx = np.argpartition([np.NINF if v is None else v for v in property_values],
                                                 -self.n_show_top)[-self.n_show_top:]

        if is_property_per_label:
            lowest_img_idx = [_sample_index_from_flatten_index(values_lengths_cumsum, x) for x in lowest_values_idx]
            highest_img_idx = [_sample_index_from_flatten_index(values_lengths_cumsum, x) for x in highest_values_idx]
        else:
            lowest_img_idx = lowest_values_idx
            highest_img_idx = highest_values_idx

        self._lowest_property_value_images[property_name] = \
            {'images': [images[x] for x in lowest_img_idx],
             'property_values': [[property_values[x]] if is_property_per_label else property_values[x]
                                 for x in lowest_values_idx],
             'labels': [[labels[x]] if is_property_per_label else labels[x] for x in lowest_values_idx]}
        self._highest_property_value_images[property_name] = \
            {'images': [images[x] for x in highest_img_idx],
             'property_values': [[property_values[x]] if is_property_per_label else property_values[x]
                                 for x in highest_values_idx],
             'labels': [[labels[x]] if is_property_per_label else labels[x] for x in highest_values_idx]}


def _ensure_property_shape(property_values, data_len, prop_name):
    """Validate the result of the property."""
    if len(property_values) != data_len:
        raise DeepchecksProcessError(f'Properties are expected to return value per image but instead got'
                                     f' {len(property_values)} values for {data_len} images for property '
                                     f'{prop_name}')

    # If the first item is list validate all items are list of numbers
    if isinstance(property_values[0], t.Sequence):
        if any((not isinstance(x, t.Sequence) for x in property_values)):
            raise DeepchecksProcessError(f'Property result is expected to be either all lists or all scalars but'
                                         f' got mix for property {prop_name}')
        if any((not _is_list_of_numbers(x) for x in property_values)):
            raise DeepchecksProcessError(f'For outliers, properties are expected to be only numeric types but'
                                         f' found non-numeric value for property {prop_name}')
    # If first value is not list, validate all items are numeric
    elif not _is_list_of_numbers(property_values):
        raise DeepchecksProcessError(f'For outliers, properties are expected to be only numeric types but'
                                     f' found non-numeric value for property {prop_name}')


def _is_list_of_numbers(l):
    return not any(i is not None and not isinstance(i, Number) for i in l)


def _sample_index_from_flatten_index(cumsum_lengths, flatten_index) -> int:
    # The cumulative sum lengths is holding the cumulative sum of properties per image, so the first index which value
    # is greater than the flatten index, is the image index.
    # for example if the sums lengths is [1, 6, 11, 13, 16, 20] and the flatten index = 6, it means this property
    # belong to the third image which is index = 2.
    return np.argwhere(cumsum_lengths > flatten_index)[0][0]


NO_IMAGES_TEMPLATE = """
<h3><b>Property "{prop_name}"</b></h3>
<div>{message}</div>
"""

HTML_TEMPLATE = """
<style>
    .{id}-container {{
        overflow-x: auto;
        display: flex;
        flex-direction: column;
        gap: 10px;
    }}
    .{id}-row {{
      display: flex;
      flex-direction: row;
      align-items: center;
      gap: 10px;
    }}
    .{id}-item {{
      display: flex;
      min-width: 200px;
      position: relative;
      word-wrap: break-word;
      align-items: center;
      justify-content: center;
    }}
    .{id}-title {{
        font-family: "Open Sans", verdana, arial, sans-serif;
        color: #2a3f5f
    }}
    /* A fix for jupyter widget which doesn't have width defined on HTML widget */
    .widget-html-content {{
        width: -moz-available;          /* WebKit-based browsers will ignore this. */
        width: -webkit-fill-available;  /* Mozilla-based browsers will ignore this. */
        width: fill-available;
    }}
</style>
<h5><b>Property "{prop_name}"</b></h5>
<div>
Total number of outliers: {count}
</div>
<div>
Non-outliers range: {lower_limit} to {upper_limit}
</div>
<div class="{id}-container">
    <div class="{id}-row">
        <h5 class="{id}-item">{prop_name}</h5>
        {values}
    </div>
    <div class="{id}-row">
        <h5 class="{id}-item">Image</h5>
        {images}
    </div>
</div>
"""
