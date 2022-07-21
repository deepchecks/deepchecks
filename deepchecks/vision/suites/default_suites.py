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
# pylint: disable=unused-argument
"""Functions for loading the default (built-in) vision suites for various validation stages.

Each function returns a new suite that is initialized with a list of checks and default conditions.
It is possible to customize these suites by editing the checks and conditions inside it after the suites' creation.
"""
import warnings
from typing import Any, Dict, List, Tuple

from ignite.metrics import Metric

from deepchecks.vision import Suite
from deepchecks.vision.checks import (ClassPerformance, ConfusionMatrixReport,  # SimilarImageLeakage,
                                      HeatmapComparison, ImageDatasetDrift, ImagePropertyDrift, ImagePropertyOutliers,
                                      ImageSegmentPerformance, LabelPropertyOutliers, MeanAveragePrecisionReport,
                                      MeanAverageRecallReport, ModelErrorAnalysis, NewLabels,
                                      PropertyLabelCorrelationChange, SimpleModelComparison, TrainTestLabelDrift,
                                      TrainTestPredictionDrift)

__all__ = ['train_test_validation', 'model_evaluation', 'full_suite', 'integrity_validation', 'data_integrity']


def train_test_validation(n_top_show: int = 5,
                          label_properties: List[Dict[str, Any]] = None,
                          image_properties: List[Dict[str, Any]] = None,
                          sample_size: int = None,
                          random_state: int = None,
                          **kwargs) -> Suite:
    """Suite for validating correctness of train-test split, including distribution, \
    integrity and leakage checks.

    List of Checks:
        .. list-table:: List of Checks
           :widths: 50 50
           :header-rows: 1

           * - Check Example
             - API Reference
           * - :ref:`plot_vision_new_labels`
             - :class:`~deepchecks.vision.checks.train_test_validation.NewLabels`
           * - :ref:`plot_vision_similar_image_leakage`
             - :class:`~deepchecks.vision.checks.train_test_validation.SimilarImageLeakage`
           * - :ref:`plot_vision_heatmap_comparison`
             - :class:`~deepchecks.vision.checks.train_test_validation.HeatmapComparison`
           * - :ref:`plot_vision_train_test_label_drift`
             - :class:`~deepchecks.vision.checks.train_test_validation.TrainTestLabelDrift`
           * - :ref:`plot_vision_image_property_drift`
             - :class:`~deepchecks.vision.checks.train_test_validation.ImagePropertyDrift`
           * - :ref:`plot_vision_image_dataset_drift`
             - :class:`~deepchecks.vision.checks.train_test_validation.ImageDatasetDrift`
           * - :ref:`plot_vision_feature_label_correlation_change`
             - :class:`~deepchecks.vision.checks.train_test_validation.PropertyLabelCorrelationChange`

    Parameters
    ----------
    n_top_show: int, default: 5
        Number of images to show for checks that show images.
    label_properties : List[Dict[str, Any]], default: None
        List of properties. Replaces the default deepchecks properties.
        Each property is a dictionary with keys 'name' (str), 'method' (Callable) and 'output_type' (str),
        representing attributes of said method. 'output_type' must be one of:
        - 'numeric' - for continuous ordinal outputs.
        - 'categorical' - for discrete, non-ordinal outputs. These can still be numbers,
          but these numbers do not have inherent value.
        For more on image / label properties, see the :ref:`vision_properties_guide`.
        - 'class_id' - for properties that return the class_id. This is used because these
          properties are later matched with the VisionData.label_map, if one was given.
    image_properties : List[Dict[str, Any]], default: None
        List of properties. Replaces the default deepchecks properties.
        Each property is a dictionary with keys 'name' (str), 'method' (Callable) and 'output_type' (str),
        representing attributes of said method. 'output_type' must be one of:
        - 'numeric' - for continuous ordinal outputs.
        - 'categorical' - for discrete, non-ordinal outputs. These can still be numbers,
          but these numbers do not have inherent value.
        For more on image / label properties, see the :ref:`vision_properties_guide`.
    sample_size : int , default: None
        Number of samples to use for checks that sample data. If none, using the default sample_size per check.
    random_state: int, default: None
        Random seed for all checks.
    **kwargs : dict
        additional arguments to pass to the checks.

    Returns
    -------
    Suite
        A Suite for validating correctness of train-test split, including distribution, \
        integrity and leakage checks.

    Examples
    --------
    >>> from deepchecks.vision.suites import train_test_validation
    >>> suite = train_test_validation(n_top_show=3, sample_size=100)
    >>> result = suite.run()
    >>> result.show()

    See Also
    --------
    :ref:`vision_classification_tutorial`
    :ref:`vision_detection_tutorial`
    """
    args = locals()
    args.pop('kwargs')
    non_none_args = {k: v for k, v in args.items() if v is not None}
    kwargs = {**non_none_args, **kwargs}

    return Suite(
        'Train Test Validation Suite',
        NewLabels(**kwargs).add_condition_new_label_ratio_less_or_equal(),
        # SimilarImageLeakage(**kwargs).add_condition_similar_images_less_or_equal(),
        HeatmapComparison(**kwargs),
        TrainTestLabelDrift(**kwargs).add_condition_drift_score_less_than(),
        ImagePropertyDrift(**kwargs).add_condition_drift_score_less_than(),
        ImageDatasetDrift(**kwargs),
        PropertyLabelCorrelationChange(**kwargs).add_condition_property_pps_difference_less_than(),
    )


def model_evaluation(alternative_metrics: Dict[str, Metric] = None,
                     area_range: Tuple[float, float] = (32**2, 96**2),
                     image_properties: List[Dict[str, Any]] = None,
                     prediction_properties: List[Dict[str, Any]] = None,
                     random_state: int = 42,
                     **kwargs) -> Suite:
    """Suite for evaluating the model's performance over different metrics, segments, error analysis, \
       comparing to baseline, and more.

    List of Checks:
        .. list-table:: List of Checks
           :widths: 50 50
           :header-rows: 1

           * - Check Example
             - API Reference
           * - :ref:`plot_vision_class_performance`
             - :class:`~deepchecks.vision.checks.model_evaluation.ClassPerformance`
           * - :ref:`plot_vision_mean_average_precision_report`
             - :class:`~deepchecks.vision.checks.model_evaluation.MeanAveragePrecisionReport`
           * - :ref:`plot_vision_mean_average_recall_report`
             - :class:`~deepchecks.vision.checks.model_evaluation.MeanAverageRecallReport`
           * - :ref:`plot_vision_train_test_prediction_drift`
             - :class:`~deepchecks.vision.checks.model_evaluation.TrainTestPredictionDrift`
           * - :ref:`plot_vision_simple_model_comparison`
             - :class:`~deepchecks.vision.checks.model_evaluation.SimpleModelComparison`
           * - :ref:`plot_vision_confusion_matrix`
             - :class:`~deepchecks.vision.checks.model_evaluation.ConfusionMatrixReport`
           * - :ref:`plot_vision_image_segment_performance`
             - :class:`~deepchecks.vision.checks.model_evaluation.ImageSegmentPerformance`
           * - :ref:`plot_vision_model_error_analysis`
             - :class:`~deepchecks.vision.checks.model_evaluation.ModelErrorAnalysis`

    Parameters
    ----------
    alternative_metrics : Dict[str, Metric], default: None
        A dictionary of metrics, where the key is the metric name and the value is an ignite.Metric object whose score
        should be used. If None are given, use the default metrics.
    area_range: tuple, default: (32**2, 96**2)
        Slices for small/medium/large buckets. (For object detection tasks only)
    image_properties : List[Dict[str, Any]], default: None
        List of properties. Replaces the default deepchecks properties.
        Each property is a dictionary with keys 'name' (str), 'method' (Callable) and 'output_type' (str),
        representing attributes of said method. 'output_type' must be one of:
        - 'numeric' - for continuous ordinal outputs.
        - 'categorical' - for discrete, non-ordinal outputs. These can still be numbers,
          but these numbers do not have inherent value.
        For more on image / label properties, see the :ref:`vision_properties_guide`.
    prediction_properties : List[Dict[str, Any]], default: None
        List of properties. Replaces the default deepchecks properties.
        Each property is a dictionary with keys 'name' (str), 'method' (Callable) and 'output_type' (str),
        representing attributes of said method. 'output_type' must be one of:
        - 'numeric' - for continuous ordinal outputs.
        - 'categorical' - for discrete, non-ordinal outputs. These can still be numbers,
          but these numbers do not have inherent value.
        For more on image / label properties, see the :ref:`vision_properties_guide`.
        - 'class_id' - for properties that return the class_id. This is used because these
          properties are later matched with the VisionData.label_map, if one was given.
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
    >>> from deepchecks.vision.suites import model_evaluation
    >>> suite = model_evaluation()
    >>> result = suite.run()
    >>> result.show()

    See Also
    --------
    :ref:`vision_classification_tutorial`
    :ref:`vision_detection_tutorial`
    """
    args = locals()
    args.pop('kwargs')
    non_none_args = {k: v for k, v in args.items() if v is not None}
    kwargs = {**non_none_args, **kwargs}

    return Suite(
        'Model Evaluation Suite',
        ClassPerformance(**kwargs).add_condition_train_test_relative_degradation_less_than(),
        MeanAveragePrecisionReport(**kwargs).add_condition_average_mean_average_precision_greater_than(),
        MeanAverageRecallReport(**kwargs),
        TrainTestPredictionDrift(**kwargs).add_condition_drift_score_less_than(),
        SimpleModelComparison(**kwargs).add_condition_gain_greater_than(),
        ConfusionMatrixReport(**kwargs),
        ImageSegmentPerformance(**kwargs).add_condition_score_from_mean_ratio_greater_than(),
        ModelErrorAnalysis(**kwargs)
    )


def integrity_validation(**kwargs) -> Suite:
    """Create a suite that is meant to validate integrity of the data.

    .. deprecated:: 0.7.0
            `integrity_validation` is deprecated and will be removed in deepchecks 0.8 version, it is replaced by
            `data_integrity` suite.
    """
    warnings.warn(
        'the integrity_validation suite is deprecated, use the data_integrity suite instead',
        DeprecationWarning
    )
    return data_integrity(**kwargs)


def data_integrity(image_properties: List[Dict[str, Any]] = None,
                   n_show_top: int = 5,
                   label_properties: List[Dict[str, Any]] = None,
                   **kwargs) -> Suite:
    """
    Create a suite that includes integrity checks.

    List of Checks:
        .. list-table:: List of Checks
           :widths: 50 50
           :header-rows: 1

           * - Check Example
             - API Reference
           * - :ref:`plot_vision_image_property_outliers`
             - :class:`~deepchecks.vision.checks.data_integrity.ImagePropertyOutliers`
           * - :ref:`plot_vision_label_property_outliers`
             - :class:`~deepchecks.vision.checks.model_evaluation.LabelPropertyOutliers`

    Parameters
    ----------
    image_properties : List[Dict[str, Any]], default: None
        List of properties. Replaces the default deepchecks properties.
        Each property is a dictionary with keys 'name' (str), 'method' (Callable) and 'output_type' (str),
        representing attributes of said method. 'output_type' must be one of:
        - 'numeric' - for continuous ordinal outputs.
        - 'categorical' - for discrete, non-ordinal outputs. These can still be numbers,
          but these numbers do not have inherent value.
        For more on image / label properties, see the :ref:`vision_properties_guide`
    n_show_top : int , default: 5
        number of samples to show from each direction (upper limit and bottom limit)
    label_properties : List[Dict[str, Any]], default: None
        List of properties. Replaces the default deepchecks properties.
        Each property is a dictionary with keys 'name' (str), 'method' (Callable) and 'output_type' (str),
        representing attributes of said method. 'output_type' must be one of:
        - 'numeric' - for continuous ordinal outputs.
        - 'categorical' - for discrete, non-ordinal outputs. These can still be numbers,
          but these numbers do not have inherent value.
        For more on image / label properties, see the :ref:`vision_properties_guide`
    **kwargs : dict
        additional arguments to pass to the checks.

    Returns
    -------
    Suite
        A suite that includes integrity checks.

    Examples
    --------
    >>> from deepchecks.vision.suites import data_integrity
    >>> suite = data_integrity()
    >>> result = suite.run()
    >>> result.show()

    See Also
    --------
    :ref:`vision_classification_tutorial`
    :ref:`vision_detection_tutorial`
    """
    args = locals()
    args.pop('kwargs')
    non_none_args = {k: v for k, v in args.items() if v is not None}
    kwargs = {**non_none_args, **kwargs}

    return Suite(
        'Data Integrity Suite',
        ImagePropertyOutliers(**kwargs),
        LabelPropertyOutliers(**kwargs)
    )


def full_suite(**kwargs) -> Suite:
    """Create a suite that includes many of the implemented checks, for a quick overview of your model and data."""
    return Suite(
        'Full Suite',
        model_evaluation(**kwargs),
        train_test_validation(**kwargs),
        data_integrity(**kwargs)
    )
