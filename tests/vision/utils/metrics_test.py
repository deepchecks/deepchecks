from hamcrest import equal_to, assert_that, calling, raises

from deepchecks.vision.datasets.detection.coco import yolo_wrapper
from deepchecks.vision.utils.metrics import task_type_check, TaskType, \
    get_default_classification_scorers
from deepchecks.vision.utils.metrics import calculate_metrics
from deepchecks.vision.utils.detection_precision_recall import DetectionPrecisionRecall
from deepchecks.vision.base import VisionDataset


def test_mnist_task_type_classification(trained_mnist, mnist_dataset_train):
    res = task_type_check(trained_mnist, mnist_dataset_train)
    assert_that(res, equal_to(TaskType.CLASSIFICATION))


def test_ssd_task_type_object(trained_yolov5_object_detection, coco_dataset):
    res = task_type_check(trained_yolov5_object_detection, coco_dataset)
    assert_that(res, equal_to(TaskType.OBJECT_DETECTION))


def test_iou(coco_dataloader, trained_yolov5_object_detection):
    dl = coco_dataloader
    model = trained_yolov5_object_detection

    res = calculate_metrics([DetectionPrecisionRecall()], VisionDataset(dl), model,
                            prediction_extract=yolo_wrapper)


def test_classification(trained_mnist, mnist_dataset_train):
    res = calculate_metrics(get_default_classification_scorers(10).values(), mnist_dataset_train, trained_mnist)
