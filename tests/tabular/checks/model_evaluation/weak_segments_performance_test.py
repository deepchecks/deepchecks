# ----------------------------------------------------------------------------
# Copyright (C) 2021-2023 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""Tests for weak segment performance check."""
import numpy as np
import pandas as pd
from hamcrest import any_of, assert_that, calling, close_to, equal_to, has_items, has_length, raises
from sklearn.metrics import f1_score, make_scorer

from deepchecks.core.errors import DeepchecksValueError
from deepchecks.tabular.checks import WeakSegmentsPerformance
from tests.base.utils import equal_condition_result


def test_segment_performance_diabetes(diabetes_split_dataset_and_model):
    # Arrange
    _, val, model = diabetes_split_dataset_and_model

    # Act
    result = WeakSegmentsPerformance().run(val, model)
    segments = result.value['weak_segments_list']

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
    segments = result.value['weak_segments_list']

    # Assert
    assert_that(segments, has_length(5))
    assert_that(max(segments.iloc[:, 0]), result.value['avg_score'])
    assert_that(segments.iloc[0, 0], close_to(-86, 0.01))
    assert_that(segments.iloc[0, 1], equal_to('s5'))


def test_segment_performance_iris_with_condition(iris_split_dataset_and_model):
    # Arrange
    _, val, model = iris_split_dataset_and_model
    check = WeakSegmentsPerformance().add_condition_segments_relative_performance_greater_than()

    # Act
    result = check.run(val, model)
    condition_result = result.conditions_results
    segments = result.value['weak_segments_list']

    # Assert
    assert_that(condition_result, has_items(
        equal_condition_result(
            is_pass=True,
            name='The relative performance of weakest segment is greater than 80% of average model performance.',
            details='Found a segment with Accuracy score of 0.75 in comparison to an average score of 0.92 in sampled '
                    'data.')
    ))
    assert_that(segments, has_length(3))
    assert_that(max(segments.iloc[:, 0]), result.value['avg_score'])
    assert_that(segments.iloc[0, 0], close_to(0.75, 0))
    assert_that(segments.iloc[0, 1], equal_to('petal length (cm)'))


def test_segment_performance_iris_with_arguments(iris_split_dataset_and_model):
    # Arrange
    _, val, model = iris_split_dataset_and_model
    loss_per_sample = pd.Series(np.arange(0, 1, 1/val.n_samples, dtype=float), index=val.data.index)
    scorer = {'f1_score': make_scorer(f1_score, average='micro')}

    # Act
    result = WeakSegmentsPerformance(loss_per_sample=loss_per_sample, alternative_scorer=scorer).run(val, model)
    segments = result.value['weak_segments_list']

    # Assert
    assert_that(segments, any_of(has_length(5), has_length(6)))
    assert_that(segments.iloc[0, 0], close_to(0.7692307692307693, 0.01))


def test_regression_categorical_features_avocado(avocado_split_dataset_and_model):
    # Arrange
    _, val, model = avocado_split_dataset_and_model

    # Act
    result = WeakSegmentsPerformance().run(val, model)
    segments = result.value['weak_segments_list']

    # Assert
    assert_that(segments, has_length(7))
    assert_that(segments[segments['Feature1'] == 'type']['Feature1 range'].iloc[0], equal_to(['organic']))
    assert_that(segments[segments['Feature1'] == 'type'].iloc[0, 0], close_to(-0.362,0.01))
    assert_that(segments.iloc[0, 0], close_to(-0.379, 0.01))


def test_classes_do_not_match_proba(kiss_dataset_and_model):
    # Arrange
    _, val, model = kiss_dataset_and_model
    check = WeakSegmentsPerformance()

    # Act & Assert
    assert_that(calling(check.run).with_args(val, model, model_classes=[1, 2, 3, 4, 5, 6, 7]),
                raises(DeepchecksValueError,
                       r'Predicted probabilities shape \(2, 3\) does not match the number of classes found in the '
                       r'labels \[1, 2, 3, 4, 5, 6, 7\]\.'))


def test_categorical_feat_target(adult_split_dataset_and_model):
    # Arrange
    _, val, model = adult_split_dataset_and_model
    val = val.sample()
    val.data['native-country'].iloc[0] = np.nan
    val.data['native-country'] = pd.Categorical(val.data['native-country'])
    val.data['income'] = pd.Categorical(val.data['income'])
    check = WeakSegmentsPerformance()

    # Act
    result = check.run(val, model)
    segments = result.value['weak_segments_list']

    # Assert
    assert_that(segments, has_length(10))
