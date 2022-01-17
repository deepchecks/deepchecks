from hamcrest import equal_to, assert_that, calling, raises

from deepchecks.errors import DeepchecksValueError
from deepchecks.vision.utils.metrics import task_type_check, ModelType


def test_mnist_task_type_classification(trained_mnist, mnist_dataset):
    res=task_type_check(trained_mnist, mnist_dataset)

    assert_that(res, equal_to(ModelType.CLASSIFICATION))


# def test_ssd_task_type_objdet(trained_ssd_object_detection, obj_detection_images):
#     res=task_type_check(trained_ssd_object_detection, obj_detection_images)
#
#     assert_that(res, equal_to(ModelType.CLASSIFICATION))