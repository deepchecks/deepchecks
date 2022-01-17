from hamcrest import equal_to, assert_that, calling, raises

from deepchecks.errors import DeepchecksValueError
from deepchecks.vision.utils.metrics import task_type_check, ModelType


def test_mnist_task_type(trained_mnist, mnist_dataset):
    res=task_type_check(trained_mnist, mnist_dataset)

    assert_that(res, equal_to(ModelType.CLASSIFICATION))
