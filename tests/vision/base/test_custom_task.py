from hamcrest import assert_that, is_, calling, raises

from deepchecks.core.errors import DeepchecksValueError
from deepchecks.vision import VisionData


def test_empty_vision_data(mnist_data_loader_train):
    # Arrange
    class CustomData(VisionData):
        pass

    # Act
    data = CustomData(mnist_data_loader_train)

    # Assert
    assert_that(
        calling(data.assert_image_formatter_valid).with_args(),
        raises(DeepchecksValueError, r'batch_to_images\(\) was not implemented, some checks will not run')
    )

    assert_that(
        calling(data.assert_label_formatter_valid).with_args(),
        raises(DeepchecksValueError, r'batch_to_labels\(\) was not implemented, some checks will not run')
    )


def test_exception_image_formatter(mnist_data_loader_train):
    # Arrange
    class CustomData(VisionData):
        def batch_to_images(self, batch):
            raise Exception('test exception')

    # Act & Assert
    assert_that(
        calling(CustomData).with_args(mnist_data_loader_train),
        raises(Exception, 'test exception')
    )


def test_exception_label_formatter(mnist_data_loader_train):
    # Arrange
    class CustomData(VisionData):
        def batch_to_labels(self, batch):
            raise Exception('test exception')

    # Act & Assert
    assert_that(
        calling(CustomData).with_args(mnist_data_loader_train),
        raises(Exception, 'test exception')
    )


def test_exception_get_classes(mnist_data_loader_train):
    # Arrange
    class CustomData(VisionData):
        def batch_to_labels(self, batch):
            return batch

        def get_classes(self, labels):
            raise Exception('test exception')

    # Act & Assert
    assert_that(
        calling(CustomData).with_args(mnist_data_loader_train),
        raises(Exception, 'test exception')
    )