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

from copy import copy

import numpy as np
import pandas as pd
from hamcrest import assert_that, calling, close_to, contains_exactly, contains_inanyorder, equal_to, raises

from deepchecks.core.errors import DeepchecksValueError
from deepchecks.vision.checks import ImagePropertyOutliers, PropertyLabelCorrelationChange
from deepchecks.vision.detection_data import DetectionData
from deepchecks.vision.utils.image_functions import crop_image
from deepchecks.vision.utils.image_properties import aspect_ratio, default_image_properties
from deepchecks.vision.utils.vision_properties import (PropertiesInputType, calc_vision_properties,
                                                       static_properties_from_df)
from deepchecks.vision.vision_data import VisionData
from tests.base.utils import equal_condition_result
from tests.vision.checks.train_test_validation.property_label_correlation_change_test import \
    get_coco_batch_to_images_with_bias_one_class


def rand_prop(batch):
    return [np.random.rand() for x in batch]


def mean_prop(batch):
    return [np.mean(x) for x in batch]


def label_prop(batch):
    return [int(np.log(int(x) + 1)) for x in batch]


def filter_bbox_prop(batch):
    return [[1, 2] for x in batch[0:5]]


def vision_props_to_static_format(indexes, vision_props):
    index_properties = dict(zip(indexes, [dict(zip(vision_props, t)) for t in zip(*vision_props.values())]))
    return index_properties


def _create_static_properties(train: VisionData, test: VisionData, image_properties, calc_bbox=True):
    static_props = []
    for vision_data in [train, test]:
        if vision_data is not None:
            static_prop = {}
            for i, batch in enumerate(vision_data):
                indexes = list(vision_data.data_loader.batch_sampler)[i]
                image_props = calc_vision_properties(vision_data.batch_to_images(batch), image_properties)
                static_image_prop = vision_props_to_static_format(indexes, image_props)
                if isinstance(vision_data, DetectionData) and calc_bbox:
                    bbox_props_list = []
                    count = 0
                    targets = []
                    for labels in vision_data.batch_to_labels(batch):
                        for label in labels:
                            label = label.cpu().detach().numpy()
                            bbox = label[1:]
                            # make sure image is not out of bounds
                            if round(bbox[2]) + min(round(bbox[0]), 0) <= 0 or \
                                    round(bbox[3]) <= 0 + min(round(bbox[1]), 0):
                                continue
                            class_id = int(label[0])
                            targets.append(vision_data.label_id_to_name(class_id))
                    for img, labels in zip(vision_data.batch_to_images(batch), vision_data.batch_to_labels(batch)):
                        imgs = []
                        for label in labels:
                            label = label.cpu().detach().numpy()
                            bbox = label[1:]
                            # make sure image is not out of bounds
                            if round(bbox[2]) + min(round(bbox[0]), 0) <= 0 or \
                                    round(bbox[3]) <= 0 + min(round(bbox[1]), 0):
                                continue
                            targets += []
                            imgs.append(crop_image(img, *bbox))
                        count += len(imgs)
                        bbox_props_list.append(calc_vision_properties(imgs, image_properties))
                    bbox_props = {k: [dic[k] for dic in bbox_props_list] for k in bbox_props_list[0]}
                    static_bbox_prop = vision_props_to_static_format(indexes, bbox_props)
                    static_prop.update({k: {PropertiesInputType.IMAGES: static_image_prop[k],
                                            PropertiesInputType.PARTIAL_IMAGES: static_bbox_prop[k]} for k in indexes})
                else:
                    static_prop.update({k: {PropertiesInputType.IMAGES: static_image_prop[k]} for k in indexes})
        else:
            static_prop = None
        static_props.append(static_prop)
    train_prop, tests_prop = static_props
    return train_prop, tests_prop


def test_image_properties_outliers(mnist_dataset_train, mnist_dataset_test):
    image_properties = [{'name': 'random', 'method': rand_prop, 'output_type': 'numerical'},
                        {'name': 'mean brightness', 'method': mean_prop, 'output_type': 'numerical'},
                        ]

    train_props, _ = _create_static_properties(mnist_dataset_train, mnist_dataset_test,
                                               image_properties)
    # make sure it doesn't use images
    mnist_dataset_train = copy(mnist_dataset_train)
    mnist_dataset_train.batch_to_images = None
    mnist_dataset_train._image_formatter_error = 'bad batch images'

    check_results = ImagePropertyOutliers().run(mnist_dataset_train, train_properties=train_props)
    assert_that(check_results.value.keys(), contains_exactly('random', 'mean brightness'))
    assert_that(check_results.value['mean brightness']['lower_limit'], close_to(6.487, 0.001))


def test_object_detection_missing_key(coco_train_visiondata, coco_test_visiondata):
    image_properties = [{'name': 'aspect_ratio', 'method': aspect_ratio, 'output_type': 'numerical'}]
    train_props, test_props = _create_static_properties(coco_train_visiondata, coco_test_visiondata,
                                                        image_properties, calc_bbox=False)
    # make sure it doesn't use images
    coco_train_visiondata = copy(coco_train_visiondata)
    coco_train_visiondata.batch_to_images = None
    coco_train_visiondata._image_formatter_error = 'bad batch images'

    # assert error is raised if no bbox properties passed in a check that calls bbox properties
    assert_that(calling(PropertyLabelCorrelationChange().run)
                .with_args(
        train_dataset=coco_train_visiondata, test_dataset=coco_test_visiondata,
        train_properties=train_props, test_properties=test_props)), \
        raises(DeepchecksValueError, 'bad batch images')


def test_object_detection_bad_prop(coco_train_visiondata, coco_test_visiondata):
    image_properties = [{'name': 'aspect_ratio', 'method': aspect_ratio, 'output_type': 'numerical'}]
    _, test_props = _create_static_properties(coco_train_visiondata, coco_test_visiondata,
                                                        image_properties, calc_bbox=False)
    # make sure it doesn't use images
    coco_train_visiondata = copy(coco_train_visiondata)
    coco_train_visiondata.batch_to_images = None
    coco_train_visiondata._image_formatter_error = 'bad batch images'

    # assert error is raised if bad properties passed in a check that calls bbox properties
    assert_that(calling(PropertyLabelCorrelationChange().run)
                .with_args(
        train_dataset=coco_train_visiondata, test_dataset=coco_test_visiondata,
        train_properties={'0': {'ahhh': 0, 'partial_images': [1, 2]}}, test_properties=test_props)), \
        raises(DeepchecksValueError, 'bad batch images')


def test_train_test_condition_pps_diff_fail_per_class(coco_train_visiondata, coco_test_visiondata, device):
    # Arrange
    train, test = coco_train_visiondata, coco_test_visiondata
    image_properties = default_image_properties
    train = copy(train)
    train.batch_to_images = get_coco_batch_to_images_with_bias_one_class(train.batch_to_labels)
    train_props, test_props = _create_static_properties(train, coco_test_visiondata,
                                                        image_properties)
    condition_value = 0.3
    check = PropertyLabelCorrelationChange(per_class=True, random_state=42
                                           ).add_condition_property_pps_difference_less_than(condition_value)
    # make sure it doesn't use images
    train.batch_to_images = None
    train._image_formatter_error = 'bad batch images'
    # Act
    result = check.run(train_dataset=train,
                       test_dataset=test, device=device, train_properties=train_props, test_properties=test_props)
    condition_result, *_ = check.conditions_decision(result)
    # Assert
    assert_that(condition_result, equal_condition_result(
        is_pass=False,
        name=f'Train-Test properties\' Predictive Power Score difference is less than {condition_value}',
        details='Properties and classes with PPS difference above threshold: {\'RMS Contrast\': {\'clock\': \'0.83\'}, '
                '\'Brightness\': {\'clock\': \'0.5\', \'teddy bear\': \'0.5\'}, \'Mean Blue Relative Intensity\': '
                '{\'clock\': \'0.33\'}}'
    ))


def test_static_properties_from_df(coco_train_visiondata, device):
    df = pd.DataFrame({'prop1': np.random.rand(coco_train_visiondata.num_samples),
                       'prop2': np.random.rand(coco_train_visiondata.num_samples),
                       'prop3': np.random.rand(coco_train_visiondata.num_samples)})
    stat_prop = static_properties_from_df(df, image_cols=('prop1', 'prop2'), label_cols=('prop3', ))

    assert_that(stat_prop[0].keys(), contains_inanyorder(PropertiesInputType.IMAGES,
                                                         PropertiesInputType.LABELS,
                                                         PropertiesInputType.PARTIAL_IMAGES,
                                                         PropertiesInputType.PREDICTIONS))
    assert_that(stat_prop[1][PropertiesInputType.IMAGES].keys(), contains_inanyorder('prop1', 'prop2'))
    assert_that(stat_prop[1][PropertiesInputType.PARTIAL_IMAGES], equal_to(None))
