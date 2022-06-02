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
from hamcrest import assert_that, calling, raises

from deepchecks.core.errors import DeepchecksValueError
from deepchecks.vision.utils.image_properties import default_image_properties
from deepchecks.vision.utils.image_properties import validate_properties as validate_image_properties
from deepchecks.vision.utils.label_prediction_properties import DEFAULT_OBJECT_DETECTION_LABEL_PROPERTIES
from deepchecks.vision.utils.label_prediction_properties import \
    validate_properties as validate_label_prediction_properties


def test_image_properties_validation():
    validate_image_properties(default_image_properties)


def test_image_properties_validation_with_instance_of_incorrect_type_provided():
    assert_that(
        calling(validate_image_properties).with_args(object()),
        detects_incorrect_type_of_input()
    )


def test_image_properties_validation_with_empty_properties_list():
    assert_that(
        calling(validate_image_properties).with_args([]),
        detects_empty_list()
    )


def test_image_properties_validation_with_unsupported_item_type():
    properties = [*default_image_properties, object()]

    assert_that(
        calling(validate_image_properties).with_args(properties),
        detects_incorrect_item_type(item_index=len(properties) - 1),
    )


def test_image_properties_validation_with_incorrect_property_dict_structure():
    property = default_image_properties[0].copy()
    property.pop('method')

    assert_that(
        calling(validate_image_properties).with_args([property]),
        detects_incorrect_property_dict_structure(property_name=property['name'])
    )


def test_image_properties_validation_with_incorrect_property_output_type():
    property = default_image_properties[0].copy()
    property['output_type'] = 'hello-world'

    assert_that(
        calling(validate_image_properties).with_args([property]),
        raises(
            DeepchecksValueError,
            r"List of properties contains next problems:\n"
            rf"\+ Property {property['name']}: field \"output_type\" must be one of \('categorical', 'numerical'\), "
            rf"instead got hello-world")
    )


# =====================================================


def test_label_prediction_properties_validation():
    validate_label_prediction_properties(DEFAULT_OBJECT_DETECTION_LABEL_PROPERTIES)


def test_label_prediction_properties_validation_with_instance_of_incorrect_type_provided():
    assert_that(
        calling(validate_label_prediction_properties).with_args(object()),
        detects_incorrect_type_of_input()
    )


def test_label_prediction_properties_validation_with_empty_properties_list():
    assert_that(
        calling(validate_label_prediction_properties).with_args([]),
        detects_empty_list()
    )


def test_label_prediction_validation_with_unsupported_item_type():
    properties = [*DEFAULT_OBJECT_DETECTION_LABEL_PROPERTIES, object()]

    assert_that(
        calling(validate_label_prediction_properties).with_args(properties),
        detects_incorrect_item_type(item_index=len(properties) - 1),
    )


def test_label_prediction_properties_validation_with_incorrect_property_dict_structure():
    property = DEFAULT_OBJECT_DETECTION_LABEL_PROPERTIES[0].copy()
    property.pop('method')

    assert_that(
        calling(validate_label_prediction_properties).with_args([property]),
        detects_incorrect_property_dict_structure(property_name=property['name'])
    )


def test_label_prediction_properties_validation_with_incorrect_property_output_type():
    property = DEFAULT_OBJECT_DETECTION_LABEL_PROPERTIES[0].copy()
    property['output_type'] = 'hello-world'

    assert_that(
        calling(validate_label_prediction_properties).with_args([property]),
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
