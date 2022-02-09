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
from hamcrest import equal_to, assert_that, calling, raises

from deepchecks.vision.datasets.detection.coco import yolo_wrapper
from deepchecks.vision.metrics_utils.metrics import calculate_metrics
from deepchecks.vision.metrics_utils.detection_precision_recall import AveragePrecision
from deepchecks.vision.utils.detection_formatters import DetectionLabelFormatter, DetectionPredictionFormatter
from deepchecks.vision.dataset import VisionData

def test_iou(coco_dataloader, trained_yolov5_object_detection):
    dl = VisionData(coco_dataloader, label_transformer=DetectionLabelFormatter(lambda x: x), num_classes=80)
    model = trained_yolov5_object_detection

    res = calculate_metrics([AveragePrecision()], VisionData(dl), model,
                            prediction_formatter=DetectionPredictionFormatter(yolo_wrapper))
    print(res)
