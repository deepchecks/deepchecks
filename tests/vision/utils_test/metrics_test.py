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

def test_default_ignite_complient(coco_test_visiondata: VisionData, trained_yolov5_object_detection):
    res = calculate_metrics([AveragePrecision()], coco_test_visiondata, trained_yolov5_object_detection,
                            prediction_formatter=DetectionPredictionFormatter(yolo_wrapper))
    assert_that(res.keys(), has_length(1))
    assert_that(res['AveragePrecision'], has_length(59))


def test_pycocotools_stuff(coco_test_visiondata: VisionData, trained_yolov5_object_detection):
    metric = AveragePrecision(return_single_value=False)
    for batch in coco_test_visiondata.get_data_loader():
        images = batch[0]
        label = coco_test_visiondata.label_transformer(batch[1])
        prediction = DetectionPredictionFormatter(yolo_wrapper)(trained_yolov5_object_detection(images))
        metric.update((prediction, label))
    res = metric.compute()[0]
    print(metric.get_val_at(res['precision'], area='all', max_dets=100))
    print(metric.get_val_at(res['precision'], iou=0.5, area='all', max_dets=100))
    print(metric.get_val_at(res['precision'], iou=0.75, area='all', max_dets=100))
    print(metric.get_val_at(res['precision'], area='small', max_dets=100))
    print(metric.get_val_at(res['precision'], area='medium', max_dets=100))
    print(metric.get_val_at(res['precision'], area='large', max_dets=100))
    print()
    print(metric.get_val_at(res['recall'], area='all', max_dets=1))
    print(metric.get_val_at(res['recall'], area='all', max_dets=10))
    print(metric.get_val_at(res['recall'], area='all', max_dets=100))
    print(metric.get_val_at(res['recall'], area='small', max_dets=100))
    print(metric.get_val_at(res['recall'], area='medium', max_dets=100))
    print(metric.get_val_at(res['recall'], area='large', max_dets=100))

    assert_that(metric.get_val_at(res['precision'], area='all', max_dets=100), close_to(0.38, 0.01))
    assert_that(metric.get_val_at(res['precision'], iou=0.5, area='all', max_dets=100), close_to(0.51, 0.01))
    assert_that(metric.get_val_at(res['precision'], iou=0.75, area='all', max_dets=100), close_to(0.41, 0.01))
    assert_that(metric.get_val_at(res['precision'], area='small', max_dets=100), close_to(0.14, 0.01))
    assert_that(metric.get_val_at(res['precision'], area='medium', max_dets=100), close_to(0.37, 0.01))
    assert_that(metric.get_val_at(res['precision'], area='large', max_dets=100), close_to(0.52, 0.01))

    assert_that(metric.get_val_at(res['recall'], area='all', max_dets=1), close_to(0.3, 0.01))
    assert_that(metric.get_val_at(res['recall'], area='all', max_dets=10), close_to(0.4, 0.01))
    assert_that(metric.get_val_at(res['recall'], area='all', max_dets=100), close_to(0.4, 0.01))
    assert_that(metric.get_val_at(res['recall'], area='small', max_dets=100), close_to(0.14, 0.01))
    assert_that(metric.get_val_at(res['recall'], area='medium', max_dets=100), close_to(0.41, 0.01))
    assert_that(metric.get_val_at(res['recall'], area='large', max_dets=100), close_to(0.53, 0.01))