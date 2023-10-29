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
"""New Labels check tests"""

from hamcrest import assert_that, equal_to, greater_than, has_entries, has_items, has_length

from deepchecks.vision.checks import NewLabels
from deepchecks.vision.datasets.classification.mnist_torch import collate_without_model as mnist_collate_without_model
from deepchecks.vision.datasets.detection.coco_torch import collate_without_model as coco_collate_without_model
from deepchecks.vision.utils.test_utils import replace_collate_fn_visiondata
from tests.base.utils import equal_condition_result


def test_object_detection_coco(coco_visiondata_train, coco_visiondata_test):
    # Act
    result = NewLabels().run(coco_visiondata_train, coco_visiondata_test)
    # Assert
    assert_that(result.value['new_labels'], has_entries({'donut': 14, 'tennis racket': 7}))
    assert_that(result.value, has_entries(all_labels_count=387))
    assert_that(result.display, has_length(greater_than(0)))


def test_object_detection_coco_without_display(coco_visiondata_train, coco_visiondata_test):
    # Act
    result = NewLabels().run(coco_visiondata_train, coco_visiondata_test, with_display=False)
    # Assert
    assert_that(result.value['new_labels'], has_entries({'donut': 14, 'tennis racket': 7}))
    assert_that(result.value, has_entries(all_labels_count=387))
    assert_that(result.display, has_length(0))


def test_object_detection_coco_with_condition(coco_visiondata_train, coco_visiondata_test):
    # Act
    check = NewLabels().add_condition_new_label_ratio_less_or_equal(0.1)
    result = check.conditions_decision(check.run(coco_visiondata_train, coco_visiondata_test))
    # Assert
    assert_that(result, has_items(
        equal_condition_result(
            is_pass=False,
            name='Percentage of new labels in the test set is less or equal to 10%',
            details=(
                '10.08% of labels found in test set were not in train set. '
                'New labels most common in test set: [\'donut\', \'tennis racket\', \'boat\']'))
    ))


def test_object_detection_coco_new_labels(coco_visiondata_train, coco_visiondata_test):
    # Arrange
    def modified_labels_collate(data):
        images, labels = coco_collate_without_model(data)
        modified_labels = []
        for idx, labels_per_image in enumerate(labels):
            if idx % 7 == 0 and labels_per_image.shape[0] > 0:
                labels_per_image[0, -1] = 8
            if idx % 5 == 0 and labels_per_image.shape[0] > 0:
                labels_per_image[0, -1] = 10
            modified_labels.append(labels_per_image)
        return {'images': images, 'labels': modified_labels}

    modified_test = replace_collate_fn_visiondata(coco_visiondata_test, modified_labels_collate)
    # Act
    result = NewLabels().run(coco_visiondata_train, modified_test)
    # Assert
    assert_that(result.value['new_labels'], has_entries({'donut': 14, 'tennis racket': 7}))
    assert_that(result.value, has_entries(all_labels_count=387))


def test_classification_mnist_with_condition(mnist_visiondata_train, mnist_visiondata_test):
    # Arrange
    check = NewLabels().add_condition_new_label_ratio_less_or_equal(0)
    # Act
    result = check.run(mnist_visiondata_train, mnist_visiondata_test)
    # Assert
    assert_that(check.conditions_decision(result), has_items(
        equal_condition_result(is_pass=True,
                               name='Percentage of new labels in the test set is less or equal to 0%',
                               details='No new labels were found in test set.')
    ))
    assert_that(result.value['new_labels'], has_length(0))


def test_classification_mnist_change_label_with_condition(mnist_visiondata_train, mnist_visiondata_test):
    # Arrange
    def modified_labels_collate(data):
        images, labels = mnist_collate_without_model(data)
        modified_labels = [x if x != 3 else -3 for x in labels]
        return {'images': images, 'labels': modified_labels}

    def collate_fn(data):
        images, labels = mnist_collate_without_model(data)
        return {'images': images, 'labels': labels}

    train = replace_collate_fn_visiondata(mnist_visiondata_train, collate_fn)
    modified_test = replace_collate_fn_visiondata(mnist_visiondata_test, modified_labels_collate)
    train.label_map.clear()
    modified_test.label_map.clear()
    check = NewLabels().add_condition_new_label_ratio_less_or_equal(0)
    # Act
    result = check.run(train, modified_test)
    # Assert
    assert_that(check.conditions_decision(result), has_items(
        equal_condition_result(is_pass=False,
                               name='Percentage of new labels in the test set is less or equal to 0%',
                               details='8% of labels found in test set were not in train set.'
                                       ' New labels most common in test set: [\'-3\']')
    ))
    assert_that(result.value, equal_to({'new_labels': {'-3': 16}, 'all_labels_count': 200}))


def test_classification_mnist_new_labels(mnist_visiondata_train, mnist_visiondata_test, device):
    # Arrange
    def modified_labels_collate(data):
        images, labels = mnist_collate_without_model(data)
        modified_labels = [x if x > 3 else -x for x in labels]
        return {'images': images, 'labels': modified_labels}

    def collate_fn(data):
        images, labels = mnist_collate_without_model(data)
        return {'images': images, 'labels': labels}

    modified_test = replace_collate_fn_visiondata(mnist_visiondata_test, modified_labels_collate)
    train = replace_collate_fn_visiondata(mnist_visiondata_train, collate_fn)
    train.label_map.clear()
    modified_test.label_map.clear()
    check = NewLabels().add_condition_new_label_ratio_less_or_equal(0)
    # Act
    result = check.run(train, modified_test)
    # Assert
    assert_that(result.value, equal_to({'new_labels': {'-1': 28, '-2': 16, '-3': 16},
                                        'all_labels_count': 200}))
