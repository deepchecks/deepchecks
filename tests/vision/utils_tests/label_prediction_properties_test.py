# ----------------------------------------------------------------------------
# Copyright (C) 2021-2023 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
import torch
from hamcrest import assert_that, contains_exactly

from deepchecks.vision.utils.label_prediction_properties import (_count_classes_by_segment_in_image,
                                                                 _count_pred_classes_by_segment_in_image,
                                                                 _get_predicted_classes_per_image_semantic_segmentation,
                                                                 _get_samples_per_class_semantic_segmentation,
                                                                 _get_segment_area, _get_segment_pred_area)


def test_segment_area():
    t1 = torch.Tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    t2 = torch.Tensor([[2, 1, 1], [1, 1, 1], [1, 1, 1]])

    batch = [t1, t2]
    res = _get_segment_area(batch)
    expected_result = [[9], [8, 1]]
    assert_that(res, contains_exactly(*expected_result))


def test_get_samples_per_class_semantic_segmentation():
    t1 = torch.Tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    t2 = torch.Tensor([[2, 1, 1], [1, 1, 1], [1, 1, 1]])

    batch = [t1, t2]
    res = _get_samples_per_class_semantic_segmentation(batch)
    expected_result = [[1], [1, 2]]
    assert_that(res, contains_exactly(*expected_result))


def test_count_classes_by_segment_in_image():
    t1 = torch.Tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    t2 = torch.Tensor([[2, 1, 1], [1, 1, 1], [1, 1, 1]])

    batch = [t1, t2]
    res = _count_classes_by_segment_in_image(batch)
    expected_result = [1, 2]
    assert_that(res, contains_exactly(*expected_result))


def test_get_segment_pred_area():
    t1 = torch.Tensor([[[0.3, 0.6, 0.1], [0.3, 0.6, 0.1], [0.3, 0.6, 0.1]]] * 3)
    t2 = torch.Tensor([[[0.3, 0.3, 0.4], [0.3, 0.6, 0.1], [0.3, 0.6, 0.1]]] * 3)

    t1 = torch.transpose(t1, 0, 2)
    t2 = torch.transpose(t2, 0, 2)

    batch = [t1, t2]
    res = _get_segment_pred_area(batch)
    expected_result = [[9], [6, 3]]
    assert_that(res, contains_exactly(*expected_result))


def test_get_samples_per_pred_class_semantic_segmentation():
    t1 = torch.Tensor([[[0.3, 0.6, 0.1], [0.3, 0.6, 0.1], [0.3, 0.6, 0.1]]] * 3)
    t2 = torch.Tensor([[[0.3, 0.3, 0.4], [0.3, 0.6, 0.1], [0.3, 0.6, 0.1]]] * 3)

    t1 = torch.transpose(t1, 0, 2)
    t2 = torch.transpose(t2, 0, 2)

    batch = [t1, t2]
    res = _get_predicted_classes_per_image_semantic_segmentation(batch)
    expected_result = [[1], [1, 2]]
    assert_that(res, contains_exactly(*expected_result))


def test_count_pred_classes_by_segment_in_image():
    t1 = torch.Tensor([[[0.3, 0.6, 0.1], [0.3, 0.6, 0.1], [0.3, 0.6, 0.1]]] * 3)
    t2 = torch.Tensor([[[0.3, 0.3, 0.4], [0.3, 0.6, 0.1], [0.3, 0.6, 0.1]]] * 3)

    t1 = torch.transpose(t1, 0, 2)
    t2 = torch.transpose(t2, 0, 2)

    batch = [t1, t2]
    res = _count_pred_classes_by_segment_in_image(batch)
    expected_result = [1, 2]
    assert_that(res, contains_exactly(*expected_result))
