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
from hamcrest import equal_to, assert_that, calling, raises, has_length

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
