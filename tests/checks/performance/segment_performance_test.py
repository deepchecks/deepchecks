"""Tests for segment performance check."""
from hamcrest import assert_that, has_entries, close_to, has_property, equal_to
from deepchecks.checks.performance.segment_performance import SegmentPerformance


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
