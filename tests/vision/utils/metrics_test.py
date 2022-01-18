from hamcrest import equal_to, assert_that, calling, raises

from deepchecks.errors import DeepchecksValueError
from deepchecks.vision.utils.metrics import task_type_check, ModelType, get_scorers_list


def test_mnist_task_type_classification(trained_mnist, mnist_dataset_train):
    res=task_type_check(trained_mnist, mnist_dataset_train)
    res = get_scorers_list(trained_mnist, mnist_dataset_train)
    assert_that(res, equal_to(ModelType.CLASSIFICATION))


def test_ssd_task_type_objdet(trained_yolov5_object_detection, obj_detection_images):
    results = trained_yolov5_object_detection(obj_detection_images)
    # res=task_type_check(trained_ssd_object_detection, obj_detection_images)

    assert_that(res, equal_to(ModelType.CLASSIFICATION))