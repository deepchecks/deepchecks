# ----------------------------------------------------------------------------
# Copyright (C) 2022 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
#
import typing as t

import numpy as np
import pandas as pd
from hamcrest import (assert_that, calling, close_to, contains_exactly, equal_to, greater_than, has_entries, has_items,
                      has_length, instance_of, raises)
from plotly.basedatatypes import BaseFigure

from deepchecks.vision.checks import ImagePropertyOutliers, LabelPropertyOutliers, PropertyLabelCorrelationChange
from deepchecks.vision.suites.default_suites import full_suite
from deepchecks.vision.utils.image_properties import aspect_ratio
from deepchecks.vision.utils.label_prediction_properties import DEFAULT_OBJECT_DETECTION_LABEL_PROPERTIES
from deepchecks.vision.utils.vision_properties import PropertiesInputType, calc_vision_properties
from deepchecks.vision.vision_data import VisionData
from tests.base.utils import equal_condition_result
from tests.common import assert_class_performance_display
from tests.conftest import get_expected_results_length, validate_suite_result


def rand_prop(batch):
    return [np.random.rand() for x in batch]


def mean_prop(batch):
    return [np.mean(x) for x in batch]


def label_prop(batch):
    return [int(np.log(int(x)+1)) for x in batch]

def vision_props_to_static_format(indexes, vision_props):
    index_properties = dict(zip(indexes, [dict(zip(vision_props, t)) for t in zip(*vision_props.values())]))
    return index_properties

def _create_static_properties(train: VisionData, test: VisionData, image_properties, label_properties):
    static_props = []
    for vision_data in [train, test]:
        if vision_data is not None:
            static_prop = {}
            for i, batch in enumerate(vision_data):
                image_props = calc_vision_properties(vision_data.batch_to_images(batch), image_properties)
                label_props = calc_vision_properties(vision_data.batch_to_labels(batch), label_properties)
                indexes = list(vision_data.data_loader.batch_sampler)[i]
                static_image_prop = vision_props_to_static_format(indexes, image_props)
                static_label_prop = vision_props_to_static_format(indexes, label_props)
                static_prop.update({k: {'images': static_image_prop[k], 'labels': static_label_prop[k]} for k in
                                    indexes})

        else:
            static_prop = None
        static_props.append(static_prop)
    train_prop, tests_prop = static_props
    return train_prop, tests_prop


def test_image_properties_outliers(mnist_dataset_train, mnist_dataset_test):
    image_properties = [{'name': 'random', 'method': rand_prop, 'output_type': 'numerical'},
                        {'name': 'mean brightness', 'method': mean_prop, 'output_type': 'numerical'},
                        ]

    label_properties = [{'name': 'log', 'method': label_prop, 'output_type': 'numerical'}]
    train_props, test_props = _create_static_properties(mnist_dataset_train, mnist_dataset_test,
                                                        image_properties, label_properties)
    check_results = ImagePropertyOutliers().run(mnist_dataset_train,train_properties=train_props)
    assert_that(check_results.value.keys(), contains_exactly('random', 'mean brightness'))
    assert_that(check_results.value['mean brightness']['lower_limit'], close_to(13.87, 0.001))


def test_object_detection(coco_train_visiondata, coco_test_visiondata):
    image_properties = [{'name': 'aspect_ratio', 'method': aspect_ratio, 'output_type': 'numerical'}]
    label_properties = DEFAULT_OBJECT_DETECTION_LABEL_PROPERTIES
    train_props, test_props = _create_static_properties(coco_train_visiondata, coco_test_visiondata,
                                                        image_properties, label_properties)

    # assert error is raised if no bbox properties passed in a check that calls bbox properties
    assert_that(calling(PropertyLabelCorrelationChange().run)
                .with_args(
        train_dataset=coco_train_visiondata, test_dataset=coco_test_visiondata,
        train_properties=train_props, test_properties=test_props)), \
    raises(KeyError)

    # assert that label properties also work for bboxes
    check_results = LabelPropertyOutliers().run(coco_train_visiondata, train_properties=train_props)
    assert_that(check_results.value.keys(),
                contains_exactly(
                    'Samples Per Class', 'Bounding Box Area (in pixels)', 'Number of Bounding Boxes Per Image'))