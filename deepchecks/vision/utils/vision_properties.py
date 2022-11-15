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
"""Module for calculating the properties used in Vision checks."""
from enum import Enum
from itertools import chain

__all__ = ['PropertiesInputType', 'validate_properties', 'static_properties_from_df']

from collections import defaultdict
from typing import Any, Dict, List, Tuple

from deepchecks.core.errors import DeepchecksValueError


class PropertiesInputType(Enum):
    """Enum containing supported task types."""

    IMAGES = 'images'
    PARTIAL_IMAGES = 'partial_images'
    LABELS = 'labels'
    PREDICTIONS = 'predictions'


def calc_vision_properties(raw_data: List, properties_list: List) -> Dict[str, list]:
    """
    Calculate the image properties for a batch of images.

    Parameters
    ----------
    raw_data : torch.Tensor
        Batch of images to transform to image properties.

    properties_list: List[Dict] , default: None
        A list of properties to calculate.

    Returns
    ------
    batch_properties: dict[str, List]
        A dict of property name, property value per sample.
    """
    batch_properties = defaultdict(list)
    for single_property in properties_list:
        property_list = single_property['method'](raw_data)
        batch_properties[single_property['name']] = property_list
    return batch_properties


def validate_properties(properties: List[Dict[str, Any]]):
    """Validate structure of measurements."""
    if not isinstance(properties, list):
        raise DeepchecksValueError(
            'Expected properties to be a list, '
            f'instead got {type(properties).__name__}'
        )

    if len(properties) == 0:
        raise DeepchecksValueError('Properties list can\'t be empty')

    expected_keys = ('name', 'method', 'output_type')
    output_types = ('categorical', 'numerical', 'class_id')

    errors = []

    for index, image_property in enumerate(properties):

        if not isinstance(image_property, dict):
            errors.append(
                f'Item #{index}: property must be of type dict, '
                f'and include keys {expected_keys}. Instead got {type(image_property).__name__}'
            )
            continue

        image_property['name'] = property_name = image_property.get('name') or f'#{index}'
        difference = sorted(set(expected_keys).difference(set(image_property.keys())))

        if len(difference) > 0:
            errors.append(
                f'Property {property_name}: dictionary must include keys {expected_keys}. '
                f'Next keys are missed {difference}'
            )
            continue

        property_output_type = image_property['output_type']

        if property_output_type not in output_types:
            errors.append(
                f'Property {property_name}: field "output_type" must be one of {output_types}, '
                f'instead got {property_output_type}'
            )

    if len(errors) > 0:
        errors = '\n+ '.join(errors)
        raise DeepchecksValueError(f'List of properties contains next problems:\n+ {errors}')

    return properties


# pylint: disable=invalid-name
STATIC_PROPERTIES_FORMAT = Dict[int, Dict[PropertiesInputType, Dict[str, Any]]]


PROPERTIES_CACHE_FORMAT = Dict[PropertiesInputType, Dict[str, List]]


def static_prop_to_cache_format(static_props: STATIC_PROPERTIES_FORMAT) -> PROPERTIES_CACHE_FORMAT:
    """
    Format a batch of static predictions to the format in the batch object cache.

    Expects the items in all the indices to have the same properties.
    """
    indices = list(static_props.keys())
    input_types = list(static_props[indices[0]].keys())
    props_cache = {input_type: {prop_name: [] for prop_name in list(static_props[indices[0]][input_type].keys())}
                   for input_type in input_types if static_props[indices[0]][input_type]}

    for input_type in input_types:
        if static_props[indices[0]][input_type]:
            for prop_name in list(static_props[indices[0]][input_type].keys()):
                prop_vals = [static_props[index][input_type][prop_name] for index in indices]
                if input_type == PropertiesInputType.PARTIAL_IMAGES:
                    prop_vals = list(chain.from_iterable(prop_vals))
                props_cache[input_type][prop_name] = prop_vals

    return props_cache


def static_properties_from_df(df,
                              image_cols: Tuple = (),
                              label_cols: Tuple = (),
                              prediction_cols: Tuple = (),
                              partial_image_cols: Tuple = ()) -> STATIC_PROPERTIES_FORMAT:
    """
    Transform the pre-calculated properties from a DataFrame to the expected dict format.

    Read more about the excepted dict format for pre-calculated properties at
    https://docs.deepchecks.com/stable/user-guide/vision/vision_properties.html

    Parameters
    ----------
    df: pd.Dataframe
        A dataframe with the pre-calculated properties per sample, the indices should match those of the samples in the
        dataset.
    image_cols: Tuple, default: ()
        The names of the columns containing the results of the properties calculated on images.
    label_cols: Tuple, default: ()
        The names of the columns containing the results of the properties calculated on labels.
    prediction_cols: Tuple, default: ()
        The names of the columns containing the results of the properties calculated on predictions.
    partial_image_cols: Tuple, default: ()
        The names of the columns containing the results of the properties calculated on partial images - images cut out
        of the original images such as bounding boxes.

    Returns
    -------
    The static properties in the format expected by the checks.
    """
    image_props = df.loc[:, image_cols].to_dict(orient='index')
    label_props = df.loc[:, label_cols].to_dict(orient='index')
    pred_props = df.loc[:, prediction_cols].to_dict(orient='index')
    pi_props = df.loc[:, partial_image_cols].to_dict(orient='index')

    static_props = {}
    for k in df.index.to_list():
        static_props[k] = {PropertiesInputType.IMAGES: image_props[k] if image_cols else None,
                           PropertiesInputType.LABELS: label_props[k] if label_cols else None,
                           PropertiesInputType.PREDICTIONS: pred_props[k] if prediction_cols else None,
                           PropertiesInputType.PARTIAL_IMAGES: pi_props[k] if partial_image_cols else None}

    return static_props
