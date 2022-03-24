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
"""Module of ImagePropertyOutliers check."""
import typing as t
from collections import defaultdict
from numbers import Number

import numpy as np

from deepchecks import CheckResult
from deepchecks.core import DatasetKind
from deepchecks.core.errors import DeepchecksProcessError, NotEnoughSamplesError
from deepchecks.utils.outliers import iqr_outliers_range
from deepchecks.utils.strings import format_number
from deepchecks.vision import SingleDatasetCheck, Context, Batch
from deepchecks.vision.utils import image_properties


__all__ = ['ImagePropertyOutliers']

from deepchecks.vision.utils.image_functions import prepare_thumbnail


class ImagePropertyOutliers(SingleDatasetCheck):
    """Find images without outliers for the given image properties.

    The check uses `IQR <https://en.wikipedia.org/wiki/Interquartile_range#Outliers>`_ to detect outliers out of the
    single dimension properties.

    Parameters
    ----------
    alternative_image_properties : List[Dict[str, Any]], default: None
        List of properties. Replaces the default deepchecks properties.
        Each property is dictionary with keys 'name' (str), 'method' (Callable) and 'output_type' (str),
        representing attributes of said method. 'output_type' must be one of 'continuous'/'discrete'
    n_show_top : int , default: 5
        number of outliers to show from each direction (upper limit and bottom limit)
    iqr_percentiles: Tuple[int], default: (25, 75)
        Two percentiles which define the IQR range
    iqr_scale: float, default: 1.5
        The scale to multiply the IQR range for the outliers detection
    """

    _THUMBNAIL_SIZE = (200, 200)

    def __init__(self,
                 alternative_image_properties: t.List[t.Dict[str, t.Any]] = None,
                 n_show_top: int = 5,
                 iqr_percentiles: t.Tuple[int] = (25, 75),
                 iqr_scale: float = 1.5,
                 **kwargs):
        super().__init__(**kwargs)
        if alternative_image_properties is not None:
            image_properties.validate_properties(alternative_image_properties)
            self.image_properties = alternative_image_properties
        else:
            self.image_properties = image_properties.default_image_properties

        self.iqr_percentiles = iqr_percentiles
        self.iqr_scale = iqr_scale
        self.n_show_top = n_show_top
        self._properties = None

    def initialize_run(self, context: Context, dataset_kind: DatasetKind):
        self._properties = defaultdict(list)

    def update(self, context: Context, batch: Batch, dataset_kind: DatasetKind):
        images = batch.images

        for single_property in self.image_properties:
            prop_name = single_property['name']
            property_list = single_property['method'](images)
            if len(property_list) != len(images):
                raise DeepchecksProcessError(f'Properties are expected to return value per image but instead got'
                                             f'{len(property_list)} values for {len(images)} images for property '
                                             f'{prop_name}')
            if any((x is not None and not isinstance(x, Number) for x in property_list)):
                raise DeepchecksProcessError(f'For outlier check properties are expected to be only numeric types but'
                                             f' found non-numeric value for property {prop_name}')
            self._properties[prop_name].extend(property_list)

    def compute(self, context: Context, dataset_kind: DatasetKind) -> CheckResult:
        data = context.get_data_by_kind(dataset_kind)
        result = {}
        images = defaultdict(list)

        for single_property in self.image_properties:
            name = single_property['name']
            # The values are in the same order as the batch order, so always keeps the same order in order to access
            # the original sample at this index location
            values = np.array(self._properties[name])
            try:
                lower_limit, upper_limit = iqr_outliers_range(values, self.iqr_percentiles, self.iqr_scale)
            except NotEnoughSamplesError:
                result[name] = 'Not enough non-null samples to calculate outliers.'
                continue

            outliers = (values < lower_limit) | (values > upper_limit)

            # sort indices by value size
            sorted_indices = values.argsort()
            bottom_n_indices = sorted_indices[:self.n_show_top]
            top_n_indices = sorted_indices[-self.n_show_top:]
            top_and_bottom = np.concatenate((bottom_n_indices, top_n_indices))

            outliers_samples = []
            for index in top_and_bottom:
                # If not an outlier then skip
                if not outliers[index]:
                    continue

                outliers_samples.append(values[index].item())
                batch = data.batch_of_index(index)
                image = data.batch_to_images(batch)[0]
                images[name].append(prepare_thumbnail(
                    image=image,
                    size=self._THUMBNAIL_SIZE,
                    copy_image=False
                ))

            result[name] = {
                'sample_values': outliers_samples,
                'count': outliers.sum(),
                'lower_limit': lower_limit,
                'upper_limit': upper_limit
            }

        # Create display
        display = []
        for property_name, info in result.items():
            # If info is string it means there was error
            if isinstance(info, str):
                html = NO_IMAGES_TEMPLATE.format(prop_name=property_name, message=info)
            elif info['count'] == 0:
                html = NO_IMAGES_TEMPLATE.format(prop_name=property_name,
                                                 message='No outliers found.')
            else:
                values_combine = ''.join([f'<p>{format_number(x)}</p>' for x in info['sample_values']])
                images_combine = ''.join(images[property_name])

                html = HTML_TEMPLATE.format(
                    prop_name=property_name,
                    values=values_combine,
                    images=images_combine,
                    count=info['count'],
                    n_of_images=len(images[property_name]),
                    lower_limit=format_number(info['lower_limit']),
                    upper_limit=format_number(info['upper_limit'])
                )

            display.append(html)

        return CheckResult(result, display=''.join(display))


NO_IMAGES_TEMPLATE = """
<h3><b>Property "{prop_name}"</b></h3>
<div>{message}</div>
"""


HTML_TEMPLATE = """
<h3><b>Property "{prop_name}"</b></h3>
<div>
Total number of outliers: {count}
</div>
<div>
Non-outliers range: {lower_limit} to {upper_limit}
</div>
<h4>Samples</h4>
<div
    style="
        overflow-x: auto;
        display: grid;
        grid-template-rows: auto 1fr 1fr;
        grid-template-columns: auto repeat({n_of_images}, 1fr);
        grid-gap: 1.5rem;
        justify-items: center;
        align-items: center;
        padding: 2rem;
        width: max-content;">
    <h5>Value</h5>
    {values}
    <h5>Image</h5>
    {images}
</div>
"""
