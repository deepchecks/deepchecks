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
# pylint: disable=inconsistent-quotes, redefined-builtin
from hamcrest import assert_that, calling, close_to, contains_exactly, raises

from deepchecks.core.errors import DeepchecksValueError
from deepchecks.vision.utils.image_properties import default_image_properties, calc_default_image_properties
from deepchecks.vision.utils.label_prediction_properties import (DEFAULT_CLASSIFICATION_LABEL_PROPERTIES,
                                                                 DEFAULT_CLASSIFICATION_PREDICTION_PROPERTIES,
                                                                 DEFAULT_OBJECT_DETECTION_LABEL_PROPERTIES,
                                                                 DEFAULT_OBJECT_DETECTION_PREDICTION_PROPERTIES,
                                                                 DEFAULT_SEMANTIC_SEGMENTATION_LABEL_PROPERTIES,
                                                                 DEFAULT_SEMANTIC_SEGMENTATION_PREDICTION_PROPERTIES)
from deepchecks.vision.utils.vision_properties import calc_vision_properties, validate_properties


def test_calc_properties(coco_train_visiondata):
    images = coco_train_visiondata.batch_to_images(next(iter(coco_train_visiondata.data_loader)))
    results = calc_vision_properties(images, default_image_properties)
    assert_that(results.keys(), contains_exactly(
        'Aspect Ratio', 'Area', 'Brightness', 'RMS Contrast',
        'Mean Red Relative Intensity', 'Mean Green Relative Intensity', 'Mean Blue Relative Intensity'))
    assert_that(sum(results['Brightness']), close_to(15.56, 0.01))


def test_calc_default_image_properties(coco_train_visiondata):
    images = coco_train_visiondata.batch_to_images(next(iter(coco_train_visiondata.data_loader)))
    results = calc_default_image_properties(images)
    assert_that(results.keys(), contains_exactly(
        'Aspect Ratio', 'Area', 'Brightness', 'RMS Contrast',
        'Mean Red Relative Intensity', 'Mean Green Relative Intensity', 'Mean Blue Relative Intensity'))
    assert_that(sum(results['Brightness']), close_to(15.56, 0.01))


def test_default_properties():
    validate_properties(default_image_properties)
    validate_properties(DEFAULT_CLASSIFICATION_LABEL_PROPERTIES)
    validate_properties(DEFAULT_OBJECT_DETECTION_LABEL_PROPERTIES)
    validate_properties(DEFAULT_SEMANTIC_SEGMENTATION_LABEL_PROPERTIES)
    validate_properties(DEFAULT_CLASSIFICATION_PREDICTION_PROPERTIES)
    validate_properties(DEFAULT_OBJECT_DETECTION_PREDICTION_PROPERTIES)
    validate_properties(DEFAULT_SEMANTIC_SEGMENTATION_PREDICTION_PROPERTIES)


def test_validate_properties_with_instance_of_incorrect_type_provided():
    assert_that(
        calling(validate_properties).with_args(object()),
        detects_incorrect_type_of_input()
    )


def test_validate_properties_with_empty_properties_list():
    assert_that(
        calling(validate_properties).with_args([]),
        detects_empty_list()
    )


def test_validate_properties_with_unsupported_item_type():
    properties = [*default_image_properties, object()]

    assert_that(
        calling(validate_properties).with_args(properties),
        detects_incorrect_item_type(item_index=len(properties) - 1),
    )


def test_validate_properties_with_incorrect_property_dict_structure():
    property = default_image_properties[0].copy()
    property.pop('method')

    assert_that(
        calling(validate_properties).with_args([property]),
        detects_incorrect_property_dict_structure(property_name=property['name'])
    )


def test_validate_properties_with_bad_name_field():
    # Arrange
    def prop(predictions):
        return [int(x[0][0]) if len(x) != 0 else 0 for x in predictions]

    alternative_measurements = [
        {'name': 'test', 'method': prop, 'output_type': 'continuous'},
        {'name234': 'test', 'method': prop, 'output_type': 'continuous'},
    ]

    # Assert
    assert_that(
        calling(validate_properties).with_args(alternative_measurements),
        raises(
            DeepchecksValueError,
            r"List of properties contains next problems:\n"
            r"\+ Property #1: dictionary must include keys \('name', 'method', 'output_type'\)\. "
            r"Next keys are missed \['name'\]")
    )


def test_validate_properties_with_incorrect_property_output_type():
    property = DEFAULT_OBJECT_DETECTION_LABEL_PROPERTIES[0].copy()
    property['output_type'] = 'hello-world'

    assert_that(
        calling(validate_properties).with_args([property]),
        raises(
            DeepchecksValueError,
            r"List of properties contains next problems:\n"
            rf"\+ Property {property['name']}: field \"output_type\" must be one "
            r"of \('categorical', 'numerical', 'class_id'\), instead got hello-world")
    )


# ===========================


def detects_empty_list():
    return raises(
        DeepchecksValueError,
        "Properties list can't be empty"
    )


def detects_incorrect_type_of_input(provided_value_type: str = "object"):
    return raises(
        DeepchecksValueError,
        fr"Expected properties to be a list, instead got {provided_value_type}"
    )


def detects_incorrect_item_type(item_index: int, provided_item_type: str = "object"):
    return raises(
        DeepchecksValueError,
        r"List of properties contains next problems:\n"
        rf"\+ Item \#{item_index}: property must be of type dict, "
        fr"and include keys \('name', 'method', 'output_type'\)\. Instead got {provided_item_type}"
    )


def detects_incorrect_property_dict_structure(property_name: str, missed_key: str = 'method'):
    return raises(
        DeepchecksValueError,
        r"List of properties contains next problems:\n"
        rf"\+ Property {property_name}: dictionary must include keys \('name', 'method', 'output_type'\)\. "
        fr"Next keys are missed \['{missed_key}'\]"
    )
