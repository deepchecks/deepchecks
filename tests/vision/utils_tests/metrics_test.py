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
import torch
from hamcrest import has_items, assert_that, has_length, close_to

from deepchecks.vision.metrics_utils.metrics import calculate_metrics
from deepchecks.vision.metrics_utils.detection_precision_recall import AveragePrecision
from deepchecks.vision.utils.detection_formatters import DetectionPredictionFormatter
from deepchecks.vision import VisionData


def test_default_ap_ignite_complient(fake_coco_test_visiondata: VisionData, fake_trained_yolov5_object_detection, simple_formatter):
    res = calculate_metrics({'AveragePrecision': AveragePrecision()},
                            fake_coco_test_visiondata, fake_trained_yolov5_object_detection,
                            prediction_formatter=DetectionPredictionFormatter(simple_formatter),
                            device=torch.device('cpu'))
    assert_that(res.keys(), has_length(1))
    assert_that(res['AveragePrecision'], has_length(59))


def test_ar_ignite_complient(fake_coco_test_visiondata: VisionData, fake_trained_yolov5_object_detection, simple_formatter):
    res = calculate_metrics({'AveragePrecision': AveragePrecision(return_option=1)},
                            fake_coco_test_visiondata, fake_trained_yolov5_object_detection,
                            prediction_formatter=DetectionPredictionFormatter(simple_formatter),
                            device=torch.device('cpu'))

    assert_that(res.keys(), has_length(1))
    assert_that(res['AveragePrecision'], has_length(59))


def test_equal_pycocotools(fake_coco_test_visiondata: VisionData, fake_trained_yolov5_object_detection, simple_formatter):
    metric = AveragePrecision(return_option=None)
    for batch in fake_coco_test_visiondata.get_data_loader():
        label = fake_coco_test_visiondata.label_formatter(batch)
        prediction = DetectionPredictionFormatter(simple_formatter)(batch, fake_trained_yolov5_object_detection,
                                                                             torch.device('cpu'))
        metric.update((prediction, label))
    res = metric.compute()[0]

    assert_that(metric.get_classes_scores_at(res['precision'], area='all', max_dets=100), close_to(0.409, 0.001))
    assert_that(metric.get_classes_scores_at(res['precision'], iou=0.5, area='all', max_dets=100),
                close_to(0.566, 0.001))
    assert_that(metric.get_classes_scores_at(res['precision'], iou=0.75, area='all', max_dets=100),
                close_to(0.425, 0.001))
    assert_that(metric.get_classes_scores_at(res['precision'], area='small', max_dets=100), close_to(0.212, 0.001))
    assert_that(metric.get_classes_scores_at(res['precision'], area='medium', max_dets=100), close_to(0.383, 0.001))
    assert_that(metric.get_classes_scores_at(res['precision'], area='large', max_dets=100), close_to(0.541, 0.001))

    assert_that(metric.get_classes_scores_at(res['recall'], area='all', max_dets=1), close_to(0.330, 0.001))
    assert_that(metric.get_classes_scores_at(res['recall'], area='all', max_dets=10), close_to(0.423, 0.001))
    assert_that(metric.get_classes_scores_at(res['recall'], area='all', max_dets=100), close_to(0.429, 0.001))
    assert_that(metric.get_classes_scores_at(res['recall'], area='small', max_dets=100), close_to(0.220, 0.001))
    assert_that(metric.get_classes_scores_at(res['recall'], area='medium', max_dets=100), close_to(0.423, 0.001))
    assert_that(metric.get_classes_scores_at(res['recall'], area='large', max_dets=100), close_to(0.549, 0.001))

    # unrelated to coco but needed to check another param
    assert_that(metric.get_classes_scores_at(res['recall'], area='large', max_dets=100, get_mean_val=False,
                zeroed_negative=False), has_items([-1]))
    assert_that(metric.get_classes_scores_at(res['recall'], get_mean_val=False, zeroed_negative=False), has_items([-1]))
