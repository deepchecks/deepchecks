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
# pylint: disable=unused-argument
"""Functions for loading the default (built-in) nlp suites for various validation stages.

Each function returns a new suite that is initialized with a list of checks and default conditions.
It is possible to customize these suites by editing the checks and conditions inside it after the suites' creation.
"""

from deepchecks.nlp import Suite
from deepchecks.nlp.checks import (ConflictingLabels, FrequentSubstrings, LabelDrift, MetadataSegmentsPerformance,
                                   PredictionDrift, PropertyDrift, PropertyLabelCorrelation,
                                   PropertySegmentsPerformance, SpecialCharacters, TextDuplicates, TextEmbeddingsDrift,
                                   TextPropertyOutliers, TrainTestPerformance, TrainTestSamplesMix,
                                   UnderAnnotatedMetaDataSegments, UnderAnnotatedPropertySegments, UnknownTokens)

__all__ = ['data_integrity', 'train_test_validation',
           'model_evaluation', 'full_suite']


def data_integrity(n_samples: int = None,
                   random_state: int = 42,
                   **kwargs) -> Suite:
    """Suite for detecting integrity issues within a single dataset.

    Parameters
    ----------
    n_samples : int , default: None
        number of samples to use for checks that sample data. If none, using the default n_samples per check.
    random_state : int, default: 42
        random seed for all checkss.
    **kwargs : dict
        additional arguments to pass to the checks.

    Returns
    -------
    Suite
        A suite for validating correctness of train-test split, including distribution, \
        leakage and integrity checks.

    Examples
    --------
    >>> from deepchecks.nlp.suites import data_integrity
    >>> suite = data_integrity(n_samples=1_000_000)
    >>> result = suite.run()
    >>> result.show()
    """
    args = locals()
    args.pop('kwargs')
    non_none_args = {k: v for k, v in args.items() if v is not None}
    kwargs = {**non_none_args, **kwargs}
    return Suite(
        'Data Integrity Suite',
        TextPropertyOutliers(**kwargs).add_condition_outlier_ratio_less_or_equal(),
        UnknownTokens(**kwargs).add_condition_ratio_of_unknown_words_less_or_equal(),
        UnderAnnotatedPropertySegments(**kwargs).add_condition_segments_relative_performance_greater_than(),
        UnderAnnotatedMetaDataSegments(**kwargs).add_condition_segments_relative_performance_greater_than(),
        PropertyLabelCorrelation(**kwargs).add_condition_property_pps_less_than(),
        ConflictingLabels(**kwargs).add_condition_ratio_of_conflicting_labels_less_or_equal(),
        TextDuplicates(**kwargs).add_condition_ratio_less_or_equal(),
        SpecialCharacters(**kwargs).add_condition_samples_ratio_w_special_characters_less_or_equal(),
        FrequentSubstrings(**kwargs).add_condition_zero_result(),

    )


def train_test_validation(n_samples: int = None,
                          random_state: int = 42,
                          **kwargs) -> Suite:
    """Suite for validating correctness of train-test split, including distribution, \
    leakage and integrity checks.

    Parameters
    ----------
    n_samples : int , default: None
        number of samples to use for checks that sample data. If none, using the default n_samples per check.
    random_state : int, default: 42
        random seed for all checkss.
    **kwargs : dict
        additional arguments to pass to the checks.

    Returns
    -------
    Suite
        A suite for validating correctness of train-test split, including distribution, \
        leakage and integrity checks.

    Examples
    --------
    >>> from deepchecks.nlp.suites import train_test_validation
    >>> suite = train_test_validation(n_samples=1_000_000)
    >>> result = suite.run()
    >>> result.show()
    """
    args = locals()
    args.pop('kwargs')
    non_none_args = {k: v for k, v in args.items() if v is not None}
    kwargs = {**non_none_args, **kwargs}
    return Suite(
        'Train Test Validation Suite',
        PropertyDrift(**kwargs).add_condition_drift_score_less_than(),
        LabelDrift(**kwargs).add_condition_drift_score_less_than(),
        TextEmbeddingsDrift(**kwargs).add_condition_overall_drift_value_less_than(),
        TrainTestSamplesMix(**kwargs).add_condition_duplicates_ratio_less_or_equal(),
    )


def model_evaluation(n_samples: int = None,
                     random_state: int = 42,
                     **kwargs) -> Suite:
    """Suite for evaluating the model's performance over different metrics, segments, error analysis, examining \
       overfitting, comparing to baseline, and more.

    Parameters
    ----------
    n_samples : int , default: 1_000_000
        number of samples to use for checks that sample data. If none, use the default n_samples per check.
    random_state : int, default: 42
        random seed for all checks.
    **kwargs : dict
        additional arguments to pass to the checks.

    Returns
    -------
    Suite
        A suite for evaluating the model's performance.

    Examples
    --------
    >>> from deepchecks.nlp.suites import model_evaluation
    >>> suite = model_evaluation(n_samples=1_000_000)
    >>> result = suite.run()
    >>> result.show()
    """
    args = locals()
    args.pop('kwargs')
    non_none_args = {k: v for k, v in args.items() if v is not None}
    kwargs = {**non_none_args, **kwargs}

    return Suite(
        'Model Evaluation Suite',
        PredictionDrift(**kwargs).add_condition_drift_score_less_than(),
        TrainTestPerformance(**kwargs).add_condition_train_test_relative_degradation_less_than(),
        PropertySegmentsPerformance(**kwargs).add_condition_segments_relative_performance_greater_than(),
        MetadataSegmentsPerformance(**kwargs).add_condition_segments_relative_performance_greater_than(),
    )


def full_suite(**kwargs) -> Suite:
    """Create a suite that includes many of the implemented checks, for a quick overview of your model and data."""
    return Suite(
        'Full Suite',
        model_evaluation(**kwargs),
        train_test_validation(**kwargs),
        data_integrity(**kwargs),
    )
