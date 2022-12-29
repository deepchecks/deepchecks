import torch
import tensorflow as tf
from tests.conftest import get_expected_results_length, validate_suite_result
from deepchecks.vision.suites.default_suites import full_suite
from deepchecks.vision.datasets.detection import coco_tensorflow, coco_torch
from deepchecks.vision.datasets.classification import mnist


def test_full_suite_execution_torch(device):
    if torch.cuda.is_available():
        mnist_train_gpu = mnist.load_dataset(train=True, object_type='VisionData', n_samples=200, device=device)
        mnist_test_gpu = mnist.load_dataset(train=False, object_type='VisionData', n_samples=200, device=device)
        mnist_iterator_train_gpu = mnist.load_dataset(train=True, use_iterable_dataset=True, object_type='VisionData',
                                                      n_samples=200, device=device)
        mnist_iterator_test_gpu = mnist.load_dataset(train=False, use_iterable_dataset=True, object_type='VisionData',
                                                     n_samples=200, device=device)
        coco_train_gpu = coco_torch.load_dataset(train=True, object_type='VisionData', n_samples=200, device=device)
        coco_test_gpu = coco_torch.load_dataset(train=False, object_type='VisionData', n_samples=200, device=device)
        suite = full_suite(imaginery_kwarg='just to make sure all checks have kwargs in the init')
        arguments = (
            dict(train_dataset=mnist_train_gpu, test_dataset=mnist_test_gpu),
            dict(train_dataset=mnist_iterator_train_gpu, test_dataset=mnist_iterator_test_gpu),
            dict(train_dataset=coco_train_gpu, test_dataset=coco_test_gpu),
        )

        for args in arguments:
            result = suite.run(**args)
            length = get_expected_results_length(suite, args)
            validate_suite_result(result, length)


def test_full_suite_execution_tensorflow():
    if len(tf.config.list_physical_devices('GPU')) > 0:
        with tf.device('/device:GPU:0'):
            coco_train_gpu = coco_tensorflow.load_dataset(train=True, object_type='VisionData', n_samples=200)
            coco_test_gpu = coco_tensorflow.load_dataset(train=False, object_type='VisionData', n_samples=200)
            suite = full_suite(imaginery_kwarg='just to make sure all checks have kwargs in the init')
            arguments = (
                dict(train_dataset=coco_train_gpu, test_dataset=coco_test_gpu),
            )

            for args in arguments:
                result = suite.run(**args)
                length = get_expected_results_length(suite, args)
                validate_suite_result(result, length)