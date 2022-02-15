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
from hamcrest import equal_to, assert_that, calling, raises, has_length, close_to
import numpy as np

from deepchecks.vision.datasets.detection.coco import yolo_wrapper
from deepchecks.vision.metrics_utils.metrics import calculate_metrics
from deepchecks.vision.metrics_utils.detection_precision_recall import AveragePrecision
from deepchecks.vision.utils.detection_formatters import DetectionPredictionFormatter
from deepchecks.vision import VisionData

def test_default_ap_ignite_complient(coco_test_visiondata: VisionData, trained_yolov5_object_detection):
    res = calculate_metrics([AveragePrecision()], coco_test_visiondata, trained_yolov5_object_detection,
                            prediction_formatter=DetectionPredictionFormatter(yolo_wrapper))
    assert_that(res.keys(), has_length(1))
    assert_that(res['AveragePrecision'], has_length(59))


def test_ar_ignite_complient(coco_test_visiondata: VisionData, trained_yolov5_object_detection):
    res = calculate_metrics([AveragePrecision(return_option=1)], coco_test_visiondata, trained_yolov5_object_detection,
                            prediction_formatter=DetectionPredictionFormatter(yolo_wrapper))
    assert_that(res.keys(), has_length(1))
    assert_that(res['AveragePrecision'], has_length(59))


def test_equal_pycocotools(coco_test_visiondata: VisionData, trained_yolov5_object_detection):
    metric = AveragePrecision(return_option=2)
    for batch in coco_test_visiondata.get_data_loader():
        images = batch[0]
        label = coco_test_visiondata.label_transformer(batch[1])
        prediction = DetectionPredictionFormatter(yolo_wrapper)(trained_yolov5_object_detection(images))
        metric.update((prediction, label))
    res = metric.compute()[0]

    assert_that(metric.get_val_at(res['precision'], area='all', max_dets=100), close_to(0.361, 0.001))
    assert_that(metric.get_val_at(res['precision'], iou=0.5, area='all', max_dets=100), close_to(0.502, 0.001))
    assert_that(metric.get_val_at(res['precision'], iou=0.75, area='all', max_dets=100), close_to(0.376, 0.001))
    assert_that(metric.get_val_at(res['precision'], area='small', max_dets=100), close_to(0.189, 0.001))
    assert_that(metric.get_val_at(res['precision'], area='medium', max_dets=100), close_to(0.367, 0.001))
    assert_that(metric.get_val_at(res['precision'], area='large', max_dets=100), close_to(0.476, 0.001))

    assert_that(metric.get_val_at(res['recall'], area='all', max_dets=1), close_to(0.3, 0.001))
    assert_that(metric.get_val_at(res['recall'], area='all', max_dets=10), close_to(0.379, 0.001))
    assert_that(metric.get_val_at(res['recall'], area='all', max_dets=100), close_to(0.388, 0.001))
    assert_that(metric.get_val_at(res['recall'], area='small', max_dets=100), close_to(0.194, 0.001))
    assert_that(metric.get_val_at(res['recall'], area='medium', max_dets=100), close_to(0.403, 0.001))
    assert_that(metric.get_val_at(res['recall'], area='large', max_dets=100), close_to(0.488, 0.001))
