from hamcrest import equal_to, assert_that, calling, raises

from deepchecks.errors import DeepchecksValueError
from deepchecks.vision.utils.metrics import task_type_check, TaskType, get_scorers_list
from deepchecks.vision.datasets.detection import coco
from deepchecks.vision.utils.metrics import calculate_metrics
from deepchecks.vision.utils.detection_precision_recall import DetectionPrecisionRecall
from deepchecks.vision.base import VisionDataset

def test_mnist_task_type_classification(trained_mnist, mnist_dataset_train):
    res=task_type_check(trained_mnist, mnist_dataset_train)
    res = get_scorers_list(trained_mnist, mnist_dataset_train)
    assert_that(res, equal_to(TaskType.CLASSIFICATION))


def test_ssd_task_type_object(trained_yolov5_object_detection, obj_detection_images):
    results = trained_yolov5_object_detection(obj_detection_images)
    # res=task_type_check(trained_ssd_object_detection, obj_detection_images)

    assert_that(results, equal_to(TaskType.CLASSIFICATION))


def test_iou():
    dl = coco.get_coco_dataloader()
    model = coco.get_trained_yolov5_object_detection()

    def process_function(_, batch):
        X = batch[0]
        Y = batch[1]

        predictions = model.forward(X).xywh

        return predictions, Y
    res = calculate_metrics([DetectionPrecisionRecall()], VisionDataset(dl), model, prediction_extract=process_function)
    res