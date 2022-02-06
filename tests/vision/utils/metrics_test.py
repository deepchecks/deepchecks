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
from deepchecks.vision.metrics_utils.metrics import TaskType, \
    get_default_classification_scorers
from deepchecks.vision.metrics_utils.metrics import calculate_metrics
from deepchecks.vision.metrics_utils.detection_precision_recall import AveragePrecision
from deepchecks.vision.dataset import VisionData


# def test_mnist_task_type_classification(trained_mnist, mnist_dataset_train):
#     res = task_type_check(trained_mnist, mnist_dataset_train)
#     assert_that(res, equal_to(TaskType.CLASSIFICATION))
#
#
# def test_ssd_task_type_object(trained_yolov5_object_detection, coco_data):
#     res = task_type_check(trained_yolov5_object_detection, coco_data)
#     assert_that(res, equal_to(TaskType.OBJECT_DETECTION))
#
#
# def test_iou(coco_dataloader, trained_yolov5_object_detection):
#     dl = coco_dataloader
#     model = trained_yolov5_object_detection
#
#     res = calculate_metrics([AveragePrecision()], VisionData(dl), model,
#                             prediction_extract=yolo_wrapper)
#
#
# def test_classification(trained_mnist, mnist_dataset_train):
#     res = calculate_metrics(get_default_classification_scorers(10).values(), mnist_dataset_train, trained_mnist)