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
"""Test for the check property label correlation."""
import numpy as np
from hamcrest import assert_that, close_to, contains_exactly, has_entries

from deepchecks.vision.checks import PropertyLabelCorrelation
from deepchecks.vision.utils.transformations import un_normalize_batch
from tests.base.utils import equal_condition_result


def mnist_batch_to_images_with_bias(batch):
    """Create function which inverse the data normalization."""
    tensor = batch[0]
    tensor = tensor.permute(0, 2, 3, 1)
    ret = un_normalize_batch(tensor, (0.1307,), (0.3081,))
    for i, label in enumerate(batch[1]):
        label = label.cpu().detach()
        ret[i] = ret[i].clip(min=5 * label, max=180 + 5 * label)
    return ret


def get_coco_batch_to_images_with_bias(label_formatter):
    def ret_func(batch):
        ret = [np.array(x) for x in batch[0]]
        for i, labels in enumerate(label_formatter(batch)):
            for label in labels:
                if label[0] > 40:
                    x, y, w, h = [round(float(n)) for n in label[1:]]
                    ret[i][y:y + h, x:x + w] = ret[i][y:y + h, x:x + w].clip(min=200)
        return ret

    return ret_func


def med_prop(batch):
    return [np.median(x) for x in batch]


def mean_prop(batch):
    return [np.mean(x) for x in batch]


def test_classification_without_bias(mnist_dataset_train, device):
    result = PropertyLabelCorrelation().add_condition_property_pps_less_than().run(mnist_dataset_train, device=device)
    # assert check result
    assert_that(result.value, has_entries({'Brightness': close_to(0.0737, 0.005), 'Area': close_to(0.0, 0.005)}))
    # assert condition
    assert_that(result.conditions_results[0], equal_condition_result(
        is_pass=True,
        details='Passed for all of the properties',
        name='Properties\' Predictive Power Score is less than 0.8'))


def test_classification_with_bias(mnist_dataset_train, device):
    mnist_dataset_train_copy = mnist_dataset_train.copy()
    mnist_dataset_train_copy.batch_to_images = mnist_batch_to_images_with_bias
    result = PropertyLabelCorrelation().add_condition_property_pps_less_than(0.2).run(
        mnist_dataset_train_copy, device=device
    )
    # assert check result
    assert_that(result.value, has_entries({'Brightness': close_to(0.234, 0.005), 'Area': close_to(0.0, 0.005)}))
    # assert condition
    assert_that(result.conditions_results[0], equal_condition_result(
        is_pass=False,
        details='Found 1 out of 7 properties with PPS above threshold: {\'Brightness\': \'0.23\'}',
        name='Properties\' Predictive Power Score is less than 0.2'))


def test_classification_with_alternative_properties(mnist_dataset_train, device):
    alt_props = [{'name': 'med', 'method': med_prop, 'output_type': 'numerical'},
                 {'name': 'mean', 'method': mean_prop, 'output_type': 'numerical'}]
    result = PropertyLabelCorrelation(image_properties=alt_props).run(mnist_dataset_train, device=device)
    assert_that(result.value.keys(), contains_exactly('mean', 'med'))
    assert_that(result.value['med'], close_to(0.0, 0.005))


def test_object_detection_without_bias(coco_train_visiondata, device):
    result = PropertyLabelCorrelation().run(coco_train_visiondata, device=device)
    assert_that(result.value, has_entries({'Brightness': close_to(0.0, 0.005), 'Area': close_to(0.0, 0.005)}))


def test_object_detection_with_bias(coco_train_visiondata, device):
    coco_train_visiondata_copy = coco_train_visiondata.copy()
    coco_train_visiondata_copy.batch_to_images = \
        get_coco_batch_to_images_with_bias(coco_train_visiondata_copy.batch_to_labels)
    result = PropertyLabelCorrelation().run(coco_train_visiondata_copy, device=device)
    assert_that(result.value, has_entries({'Brightness': close_to(0.0459, 0.005), 'Area': close_to(0.0, 0.005)}))

