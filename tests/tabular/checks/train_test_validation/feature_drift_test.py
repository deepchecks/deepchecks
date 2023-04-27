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
"""Test functions of drift."""
import numpy as np
import pandas as pd
from hamcrest import (assert_that, calling, close_to, equal_to, greater_than, has_entries, has_item, has_length, is_not,
                      raises)

from deepchecks.core.errors import NotEnoughSamplesError
from deepchecks.tabular.checks import FeatureDrift
from tests.base.utils import equal_condition_result


def test_drift_with_model(drifted_data_and_model):
    # Arrange
    train, test, model = drifted_data_and_model
    check = FeatureDrift(categorical_drift_method='PSI', numerical_drift_method='EMD',
                                  max_num_categories=10, min_category_size_ratio=0)

    # Act
    result = check.run(train, test, model)
    # Assert
    assert_that(result.value, has_entries({
        'numeric_without_drift': has_entries(
            {'Drift score': close_to(0.01, 0.01),
             'Method': equal_to('Earth Mover\'s Distance'),
             'Importance': close_to(0.69, 0.01)}
        ),
        'numeric_with_drift': has_entries(
            {'Drift score': close_to(0.34, 0.01),
             'Method': equal_to('Earth Mover\'s Distance'),
             'Importance': close_to(0.31, 0.01)}
        ),
        'categorical_without_drift': has_entries(
            {'Drift score': close_to(0, 0.01),
             'Method': equal_to('PSI'),
             'Importance': close_to(0, 0.01)}
        ),
        'categorical_with_drift': has_entries(
            {'Drift score': close_to(0.22, 0.01),
             'Method': equal_to('PSI'),
             'Importance': close_to(0, 0.01)}
        ),
    }))
    assert_that(result.display, has_length(greater_than(0)))


def test_drift_with_model_n_top(drifted_data_and_model):
    # Arrange
    train, test, model = drifted_data_and_model
    check = FeatureDrift(categorical_drift_method='PSI', numerical_drift_method='KS', columns=[
                                  'categorical_with_drift'], n_top_columns=1, max_num_categories=10, min_category_size_ratio=0)

    # Act
    result = check.run(train, test, model)
    # Assert
    assert_that(result.value, has_entries({
        'categorical_with_drift': has_entries(
            {'Drift score': close_to(0.22, 0.01),
             'Method': equal_to('PSI'),
             'Importance': close_to(0, 0.01)}
        ),
    }))
    assert_that(result.display, has_length(4))


def test_drift_with_different_sort(drifted_data_and_model):
    # Arrange
    train, test, model = drifted_data_and_model

    # Act
    check = FeatureDrift(categorical_drift_method='PSI', sort_feature_by='feature importance')
    result = check.run(train, test, model)
    fi_display = result.display

    check = FeatureDrift(categorical_drift_method='PSI', sort_feature_by='drift + importance')
    result = check.run(train, test, model)
    sum_display = result.display

    check = FeatureDrift(categorical_drift_method='PSI', sort_feature_by='drift + importance')
    result = check.run(train, test)
    no_model_drift_display = result.display

    # Assert
    assert_that(fi_display[0], is_not(equal_to(sum_display[0])))
    assert_that(sum_display[0], is_not(equal_to(no_model_drift_display[0])))
    assert_that(fi_display[0], is_not(equal_to(no_model_drift_display[0])))


def test_drift_with_nulls(drifted_data_with_nulls):
    # Arrange
    train, test = drifted_data_with_nulls

    # Cramer's V with ignore_na=True:

    # Act
    check = FeatureDrift(numerical_drift_method='EMD')
    result = check.run(train, test)
    # Assert
    assert_that(result.value, has_entries({
        'numeric_without_drift': has_entries(
            {'Drift score': close_to(0.02, 0.01),
             'Method': equal_to('Earth Mover\'s Distance'),
             'Importance': equal_to(None)}
        ),
        'numeric_with_drift': has_entries(
            {'Drift score': close_to(0.35, 0.01),
             'Method': equal_to('Earth Mover\'s Distance'),
             'Importance': equal_to(None)}
        ),
        'categorical_without_drift': has_entries(
            {'Drift score': close_to(0.02, 0.01),
             'Method': equal_to('Cramer\'s V'),
             'Importance': equal_to(None)}
        ),
        'categorical_with_drift': has_entries(
            {'Drift score': close_to(0.23, 0.01),
             'Method': equal_to('Cramer\'s V'),
             'Importance': equal_to(None)}
        ),
    }))
    assert_that(result.display, has_length(greater_than(0)))

    # PSI with ignore_na=True:

    # Act
    check = FeatureDrift(categorical_drift_method='PSI', numerical_drift_method='EMD',
                                  max_num_categories=10, min_category_size_ratio=0)
    result = check.run(train, test)
    # Assert
    assert_that(result.value, has_entries({
        'numeric_without_drift': has_entries(
            {'Drift score': close_to(0.02, 0.01),
             'Method': equal_to('Earth Mover\'s Distance'),
             'Importance': equal_to(None)}
        ),
        'numeric_with_drift': has_entries(
            {'Drift score': close_to(0.35, 0.01),
             'Method': equal_to('Earth Mover\'s Distance'),
             'Importance': equal_to(None)}
        ),
        'categorical_without_drift': has_entries(
            {'Drift score': close_to(0, 0.01),
             'Method': equal_to('PSI'),
             'Importance': equal_to(None)}
        ),
        'categorical_with_drift': has_entries(
            {'Drift score': close_to(0.22, 0.01),
             'Method': equal_to('PSI'),
             'Importance': equal_to(None)}
        ),
    }))
    assert_that(result.display, has_length(greater_than(0)))

    # Cramer's V with ignore_na=False:

    # Act
    check = FeatureDrift(ignore_na=False, numerical_drift_method='EMD')
    result = check.run(train, test)
    # Assert
    assert_that(result.value, has_entries({
        'numeric_without_drift': has_entries(
            {'Drift score': close_to(0.02, 0.01),
             'Method': equal_to('Earth Mover\'s Distance'),
             'Importance': equal_to(None)}
        ),
        'numeric_with_drift': has_entries(
            {'Drift score': close_to(0.35, 0.01),
             'Method': equal_to('Earth Mover\'s Distance'),
             'Importance': equal_to(None)}
        ),
        'categorical_without_drift': has_entries(
            {'Drift score': close_to(0.09, 0.01),
             'Method': equal_to('Cramer\'s V'),
             'Importance': equal_to(None)}
        ),
        'categorical_with_drift': has_entries(
            {'Drift score': close_to(0.24, 0.01),
             'Method': equal_to('Cramer\'s V'),
             'Importance': equal_to(None)}
        ),
    }))
    assert_that(result.display, has_length(greater_than(0)))

    # PSI with ignore_na=False:

    # Act
    check = FeatureDrift(categorical_drift_method='PSI', numerical_drift_method='EMD',
                                  max_num_categories=10, min_category_size_ratio=0,
                                  ignore_na=False)
    result = check.run(train, test)
    # Assert
    assert_that(result.value, has_entries({
        'numeric_without_drift': has_entries(
            {'Drift score': close_to(0.02, 0.01),
             'Method': equal_to('Earth Mover\'s Distance'),
             'Importance': equal_to(None)}
        ),
        'numeric_with_drift': has_entries(
            {'Drift score': close_to(0.35, 0.01),
             'Method': equal_to('Earth Mover\'s Distance'),
             'Importance': equal_to(None)}
        ),
        'categorical_without_drift': has_entries(
            {'Drift score': close_to(0.04, 0.01),
             'Method': equal_to('PSI'),
             'Importance': equal_to(None)}
        ),
        'categorical_with_drift': has_entries(
            {'Drift score': close_to(0.24, 0.01),
             'Method': equal_to('PSI'),
             'Importance': equal_to(None)}
        ),
    }))
    assert_that(result.display, has_length(greater_than(0)))

def test_not_enough_samples(drifted_data):
    # Arrange
    train, test = drifted_data
    check = FeatureDrift(min_samples=1001)

    # Assert
    assert_that(calling(check.run).with_args(train, test),
                raises(NotEnoughSamplesError))

def test_not_enough_samples_in_columns(drifted_data_with_nulls):
    # Arrange
    train, test = drifted_data_with_nulls
    check = FeatureDrift(min_samples=999) # Data has 1000 samples but all columns have nulls

    # Assert
    assert_that(calling(check.run).with_args(train, test),
                raises(NotEnoughSamplesError))


def test_drift_with_nulls_not_enough_samples_in_numerical_columns(drifted_data_with_nulls):
    # Arrange
    train, test = drifted_data_with_nulls

    # Cramer's V with ignore_na=True:

    # Act
    check = FeatureDrift(min_samples=999, ignore_na=False)
    result = check.run(train, test)

    assert_that(result.value['numeric_with_drift']['Drift score'], equal_to(None))
    assert_that(result.display[3],
                equal_to("<span>The following columns do not have enough samples to calculate drift score: "
                         "['numeric_without_drift', 'numeric_with_drift']</span>"))

def test_reduce_output_with_nones(drifted_data_with_nulls, drifted_data_and_model):
    # Arrange

    _, _, model = drifted_data_and_model
    train, test = drifted_data_with_nulls
    check = FeatureDrift(min_samples=999, ignore_na=False, aggregation_method='l3_weighted')
    fi = pd.Series(np.ones(len(train.features)), index=train.features) / len(train.features)

    # Act
    aggregated_result = check.run(train, test, model, feature_importance=fi).reduce_output()

    # Assert
    assert_that(aggregated_result.keys(), has_item('L3 Weighted Drift Score'))
    assert_that(aggregated_result['L3 Weighted Drift Score'], close_to(0.152, 0.01))


def test_weighted_aggregation_drift_with_model(drifted_data_and_model):
    # Arrange
    train, test, model = drifted_data_and_model
    check = FeatureDrift(categorical_drift_method='PSI', numerical_drift_method='EMD',
                                  aggregation_method='weighted')

    # Act
    aggregated_result = check.run(train, test, model).reduce_output()
    # Assert
    assert_that(aggregated_result.keys(), has_item('Weighted Drift Score'))
    assert_that(aggregated_result['Weighted Drift Score'], close_to(0.1195, 0.01))


def test_aggregation_drift_with_model(drifted_data_and_model):
    # Arrange
    train, test, model = drifted_data_and_model
    check_l1 = FeatureDrift(categorical_drift_method='PSI', numerical_drift_method='EMD',
                                     max_num_categories=10, min_category_size_ratio=0,
                                     aggregation_method='weighted')
    check_l2 = FeatureDrift(categorical_drift_method='PSI', numerical_drift_method='EMD',
                                  max_num_categories=10, min_category_size_ratio=0,
                                  aggregation_method='l3_weighted')
    check_l3 = FeatureDrift(categorical_drift_method='PSI', numerical_drift_method='EMD',
                                  max_num_categories=10, min_category_size_ratio=0,
                                  aggregation_method='l5_weighted')
    # Act
    aggregated_result_l1 = check_l1.run(train, test, model).reduce_output()
    aggregated_result_l2 = check_l2.run(train, test, model).reduce_output()
    aggregated_result_l3 = check_l3.run(train, test, model).reduce_output()
    # Assert
    assert_that(aggregated_result_l2.keys(), has_item('L3 Weighted Drift Score'))
    assert_that(aggregated_result_l3.keys(), has_item('L5 Weighted Drift Score'))
    assert_that(aggregated_result_l3['L5 Weighted Drift Score'],
                greater_than(aggregated_result_l2['L3 Weighted Drift Score']))
    assert_that(aggregated_result_l1.keys(), has_item('Weighted Drift Score'))
    assert_that(aggregated_result_l2['L3 Weighted Drift Score'],
                greater_than(aggregated_result_l1['Weighted Drift Score']))

def test_none_aggregation_drift_with_model(drifted_data_and_model):
    # Arrange
    train, test, model = drifted_data_and_model
    check = FeatureDrift(categorical_drift_method='PSI', numerical_drift_method='EMD',
                                  aggregation_method=None)

    # Act
    aggregated_result = check.run(train, test, model).reduce_output()
    # Assert
    assert_that(aggregated_result.keys(), has_length(4))
    assert_that(aggregated_result.keys(), has_item('numeric_with_drift'))
    assert_that(aggregated_result['numeric_with_drift'], close_to(0.343, 0.01))


def test_weighted_aggregation_drift_no_model(drifted_data_and_model):
    # Arrange
    train, test, model = drifted_data_and_model
    check = FeatureDrift(categorical_drift_method='PSI',
                                  numerical_drift_method='EMD',
                                  aggregation_method='mean',
                                  max_num_categories=10,
                                  min_category_size_ratio=0)
    # Act
    aggregated_result = check.run(train, test).reduce_output()
    # Assert
    assert_that(aggregated_result.keys(), has_length(1))
    assert_that(aggregated_result.keys(), has_item('Mean Drift Score'))
    assert_that(aggregated_result['Mean Drift Score'], close_to(0.1475, 0.01))


def test_drift_with_model_without_display(drifted_data_and_model):
    # Arrange
    train, test, model = drifted_data_and_model
    check = FeatureDrift(categorical_drift_method='PSI', numerical_drift_method='EMD',
                                  max_num_categories=10, min_category_size_ratio=0)

    # Act
    result = check.run(train, test, model, with_display=False)
    # Assert
    assert_that(result.value, has_entries({
        'numeric_without_drift': has_entries(
            {'Drift score': close_to(0.01, 0.01),
             'Method': equal_to('Earth Mover\'s Distance'),
             'Importance': close_to(0.69, 0.01)}
        ),
        'numeric_with_drift': has_entries(
            {'Drift score': close_to(0.34, 0.01),
             'Method': equal_to('Earth Mover\'s Distance'),
             'Importance': close_to(0.31, 0.01)}
        ),
        'categorical_without_drift': has_entries(
            {'Drift score': close_to(0, 0.01),
             'Method': equal_to('PSI'),
             'Importance': close_to(0, 0.01)}
        ),
        'categorical_with_drift': has_entries(
            {'Drift score': close_to(0.22, 0.01),
             'Method': equal_to('PSI'),
             'Importance': close_to(0, 0.01)}
        ),
    }))
    assert_that(result.display, has_length(0))


def test_drift_no_model(drifted_data_and_model):
    # Arrange
    train, test, _ = drifted_data_and_model
    check = FeatureDrift(categorical_drift_method='PSI', numerical_drift_method='EMD',
                                  max_num_categories=10, min_category_size_ratio=0)

    # Act
    result = check.run(train, test)

    # Assert
    assert_that(result.value, has_entries({
        'numeric_without_drift': has_entries(
            {'Drift score': close_to(0.01, 0.01),
             'Method': equal_to('Earth Mover\'s Distance'),
             'Importance': equal_to(None)}
        ),
        'numeric_with_drift': has_entries(
            {'Drift score': close_to(0.34, 0.01),
             'Method': equal_to('Earth Mover\'s Distance'),
             'Importance': equal_to(None)}
        ),
        'categorical_without_drift': has_entries(
            {'Drift score': close_to(0, 0.01),
             'Method': equal_to('PSI'),
             'Importance': equal_to(None)}
        ),
        'categorical_with_drift': has_entries(
            {'Drift score': close_to(0.22, 0.01),
             'Method': equal_to('PSI'),
             'Importance': equal_to(None)}
        ),
    }))


def test_drift_max_drift_score_condition_fail(drifted_data_and_model):
    # Arrange
    train, test, model = drifted_data_and_model
    check = FeatureDrift(categorical_drift_method='PSI', numerical_drift_method='EMD',
                                  max_num_categories=10, min_category_size_ratio=0) \
        .add_condition_drift_score_less_than()

    # Act
    result = check.run(train, test, model)
    condition_result, *_ = result.conditions_results

    # Assert
    assert_that(condition_result, equal_condition_result(
        is_pass=False,
        name='categorical drift score < 0.2 and numerical drift score < 0.2',
        details='Failed for 2 out of 4 columns.\n'
                'Found 1 categorical columns with PSI above threshold: {\'categorical_with_drift\': \'0.22\'}\n'
                'Found 1 numeric columns with Earth Mover\'s Distance above threshold: '
                '{\'numeric_with_drift\': \'0.34\'}'
    ))


def test_drift_max_drift_score_condition_fail_cramer_and_ks(drifted_data_and_model):
    # Arrange
    train, test, model = drifted_data_and_model
    check = FeatureDrift(categorical_drift_method='cramers_v', numerical_drift_method='KS').add_condition_drift_score_less_than()

    # Act
    result = check.run(train, test, model)
    condition_result, *_ = result.conditions_results

    # Assert
    assert_that(condition_result, equal_condition_result(
        is_pass=False,
        name='categorical drift score < 0.2 and numerical drift score < 0.2',
        details='Failed for 2 out of 4 columns.\n'
                'Found 1 categorical columns with Cramer\'s V above threshold: {\'categorical_with_drift\': \'0.23\'}\n'
                'Found 1 numeric columns with Kolmogorov-Smirnov above threshold: '
                '{\'numeric_with_drift\': \'0.7\'}'
    ))


def test_drift_max_drift_score_condition_pass_threshold(drifted_data_and_model):
    # Arrange
    train, test, model = drifted_data_and_model
    check = FeatureDrift(categorical_drift_method='PSI', numerical_drift_method='EMD',
                                  max_num_categories=10, min_category_size_ratio=0) \
        .add_condition_drift_score_less_than(max_allowed_categorical_score=1,
                                             max_allowed_numeric_score=1)

    # Act
    result = check.run(train, test, model)
    condition_result, *_ = result.conditions_results

    # Assert
    assert_that(condition_result, equal_condition_result(
        is_pass=True,
        details='Passed for 4 columns out of 4 columns.\n'
                'Found column "categorical_with_drift" has the highest categorical drift score: 0.22\n'
                'Found column "numeric_with_drift" has the highest numerical drift score: 0.34',
        name='categorical drift score < 1 and numerical drift score < 1'
    ))


def test_drift_max_drift_score_multi_columns_drift_pass(drifted_data_and_model):
    # Arrange
    train, test, model = drifted_data_and_model
    check = FeatureDrift(categorical_drift_method='PSI', numerical_drift_method='EMD',
                                  max_num_categories=10, min_category_size_ratio=0) \
        .add_condition_drift_score_less_than(allowed_num_features_exceeding_threshold=2)
    # Act
    result = check.run(train, test, model)
    condition_result, *_ = result.conditions_results

    # Assert
    assert_that(condition_result, equal_condition_result(
        is_pass=True,
        details='Passed for 2 columns out of 4 columns.\n'
                'Found column "categorical_with_drift" has the highest categorical drift score: 0.22\n'
                'Found column "numeric_with_drift" has the highest numerical drift score: 0.34',
        name='categorical drift score < 0.2 and numerical drift score < 0.2'
    ))
