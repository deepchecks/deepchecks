"""
Contains unit tests for the dataset_info check
"""
from mlchecks.checks.overview.dataset_info import *
from hamcrest import *
from mlchecks.utils import MLChecksValueError


def test_assert_dataset_info(iris_dataset):
    result = dataset_info(iris_dataset)
    assert_that(result.value, equal_to((150, 5)))


def test_dataset_wrong_input():
    X = "wrong_input"
    assert_that(calling(dataset_info).with_args(X),
                raises(MLChecksValueError, 'dataset_info check must receive a DataFrame or a Dataset object'))


def test_dataset_info_object(iris_dataset):
    di = DatasetInfo()
    result = di.run(iris_dataset, model=None)
    assert_that(result.value, equal_to((150, 5)))


def test_dataset_info_dataframe(iris):
    result = dataset_info(iris)
    assert_that(result.value, equal_to((150, 5)))