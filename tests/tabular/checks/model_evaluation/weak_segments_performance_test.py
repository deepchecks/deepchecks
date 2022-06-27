# ----------------------------------------------------------------------------
# Copyright (C) 2021-2022 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""Tests for weak segment performance check."""
import pandas as pd
from hamcrest import assert_that, close_to, equal_to, has_items, has_length
from sklearn.metrics import make_scorer, f1_score

from deepchecks.tabular.checks import WeakSegmentsPerformance
from tests.base.utils import equal_condition_result


def test_segment_performance_diabetes(diabetes_split_dataset_and_model):
    # Arrange
    _, val, model = diabetes_split_dataset_and_model

    # Act
    result = WeakSegmentsPerformance().run(val, model)
    segments = result.value['segments']

    # Assert
    assert_that(segments, has_length(10))
    assert_that(max(segments.iloc[:, 0]), result.value['avg_score'])
    assert_that(segments.iloc[0, 0], close_to(-89, 1))
    assert_that(segments.iloc[0, 1], equal_to('bmi'))


def test_segment_performance_diabetes_with_arguments(diabetes_split_dataset_and_model):
    # Arrange
    _, val, model = diabetes_split_dataset_and_model

    # Act
    result = WeakSegmentsPerformance(n_top_features=4, segment_minimum_size_ratio=0.1).run(val, model)
    segments = result.value['segments']

    # Assert
    assert_that(segments, has_length(6))
    assert_that(max(segments.iloc[:, 0]), result.value['avg_score'])
    assert_that(segments.iloc[0, 0], close_to(-86, 0.01))
    assert_that(segments.iloc[0, 1], equal_to('s5'))


def test_segment_performance_iris_with_condition(iris_split_dataset_and_model):
    # Arrange
    _, val, model = iris_split_dataset_and_model
    check = WeakSegmentsPerformance().add_condition_segments_performance_relative_difference_greater_than()

    # Act
    result = check.run(val, model)
    condition_result = result.conditions_results
    segments = result.value['segments']

    # Assert
    assert_that(condition_result, has_items(
        equal_condition_result(
            is_pass=True,
            name='The performance of weakest segment is greater than 80% of average model performance.',
            details='Found a segment with Accuracy score of 0.75 in comparison to an average score of 0.92 in sampled '
                    'data.')
    ))
    assert_that(segments, has_length(6))
    assert_that(max(segments.iloc[:, 0]), result.value['avg_score'])
    assert_that(segments.iloc[0, 0], close_to(0.75, 0))
    assert_that(segments.iloc[0, 1], equal_to('petal width (cm)'))


def test_segment_performance_iris_with_arguments(iris_split_dataset_and_model):
    # Arrange
    _, val, model = iris_split_dataset_and_model
    loss_per_sample = pd.Series([1]*val.n_samples, index=val.data.index)
    scorer = {'f1_score': make_scorer(f1_score, average='micro')}

    # Act
    result = WeakSegmentsPerformance(loss_per_sample=loss_per_sample, alternative_scorer=scorer).run(val, model)
    segments = result.value['segments']

    # Assert
    assert_that(segments, has_length(6))
    assert_that(segments.iloc[0, 0], close_to(result.value['avg_score'], 0))