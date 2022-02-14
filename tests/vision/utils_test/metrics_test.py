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
    metric = AveragePrecision(get_per_class=False, return_single_value=False)
    for batch in coco_test_visiondata.get_data_loader():
        images = batch[0]
        label = coco_test_visiondata.label_transformer(batch[1])
        prediction = DetectionPredictionFormatter(yolo_wrapper)(trained_yolov5_object_detection(images))
        metric.update((prediction, label))
    res = metric.compute()[0]
    assert_that(res['precision'][-1]['all'][100][0], close_to(0.38, 0.021))
    assert_that(res['precision'][0.5]['all'][100][0], close_to(0.51, 0.01))
    assert_that(res['precision'][0.75]['all'][100][0], close_to(0.41, 0.023))
    assert_that(res['precision'][-1]['small'][100][0], close_to(0.14, 0.01))
    assert_that(res['precision'][-1]['medium'][100][0], close_to(0.37, 0.02))
    assert_that(res['precision'][-1]['large'][100][0], close_to(0.52, 0.05))
    assert_that(res['recall'][-1]['all'][1][0], close_to(0.3, 0.1))
    assert_that(res['recall'][-1]['all'][10][0], close_to(0.4, 0.014))
    assert_that(res['recall'][-1]['all'][100][0], close_to(0.4, 0.01))
    assert_that(res['recall'][-1]['small'][100][0], close_to(0.14, 0.04))
    assert_that(res['recall'][-1]['medium'][100][0], close_to(0.41, 0.03))
    assert_that(res['recall'][-1]['large'][100][0], close_to(0.53, 0.1))