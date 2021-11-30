"""Tests for segment performance check."""
from hamcrest import assert_that, has_entries, close_to, has_property, equal_to, calling, raises

from deepchecks.errors import DeepchecksValueError
from deepchecks.checks.performance.segment_performance import SegmentPerformance


def test_dataset_wrong_input():
    bad_dataset = 'wrong_input'
    # Act & Assert
    assert_that(calling(SegmentPerformance().run).with_args(bad_dataset, None),
                raises(DeepchecksValueError,
                       'Check SegmentPerformance requires dataset to be of type Dataset. instead got: str'))


def test_dataset_no_label(iris_dataset, iris_adaboost):
    # Assert
    assert_that(calling(SegmentPerformance().run).with_args(iris_dataset, iris_adaboost),
                raises(DeepchecksValueError, 'Check SegmentPerformance requires dataset to have a label column'))


def test_segment_performance_diabetes(diabetes_split_dataset_and_model):
    # Arrange
    _, val, model = diabetes_split_dataset_and_model

    # Act
    result = SegmentPerformance(feature_1='age', feature_2='sex').run(val, model).value

    # Assert
    assert_that(result, has_entries({
        'scores': has_property('shape', (10, 2)),
        'counts': has_property('shape', (10, 2))
    }))
    assert_that(result['scores'].mean(), close_to(-53, 1))
    assert_that(result['counts'].sum(), equal_to(146))


def test_segment_top_features(diabetes_split_dataset_and_model):
    # Arrange
    _, val, model = diabetes_split_dataset_and_model

    # Act
    result = SegmentPerformance().run(val, model).value

    # Assert
    assert_that(result, has_entries({
        'scores': has_property('shape', (10, 10)),
        'counts': has_property('shape', (10, 10))
    }))
    assert_that(result['counts'].sum(), equal_to(146))
