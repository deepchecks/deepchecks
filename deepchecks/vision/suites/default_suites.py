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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from deepchecks.vision import Suite
from deepchecks.vision.checks import (ClassPerformance, ConfusionMatrixReport,  # SimilarImageLeakage,
                                      HeatmapComparison, ImageDatasetDrift, ImagePropertyDrift, ImagePropertyOutliers,
                                      LabelPropertyOutliers, MeanAveragePrecisionReport, MeanAverageRecallReport,
                                      NewLabels, PropertyLabelCorrelation, PropertyLabelCorrelationChange,
                                      SimpleModelComparison, TrainTestLabelDrift, TrainTestPredictionDrift,
                                      WeakSegmentsPerformance)

__all__ = ['train_test_validation', 'model_evaluation', 'full_suite', 'data_integrity']


def train_test_validation(label_properties: List[Dict[str, Any]] = None, image_properties: List[Dict[str, Any]] = None,
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
    label_properties : List[Dict[str, Any]], default: None
        List of properties. Replaces the default deepchecks properties.
        Each property is a dictionary with keys ``'name'`` (str), ``method`` (Callable) and ``'output_type'`` (str),
        representing attributes of said method. 'output_type' must be one of:

        - ``'numeric'`` - for continuous ordinal outputs.
        - ``'categorical'`` - for discrete, non-ordinal outputs. These can still be numbers,
          but these numbers do not have inherent value.
        - ``'class_id'`` - for properties that return the class_id. This is used because these
          properties are later matched with the ``VisionData.label_map``, if one was given.

        For more on image / label properties, see the guide about :ref:`vision_properties_guide`.

    image_properties : List[Dict[str, Any]], default: None
        List of properties. Replaces the default deepchecks properties.
        Each property is a dictionary with keys ``'name'`` (str), ``method`` (Callable) and ``'output_type'`` (str),
        representing attributes of said method. 'output_type' must be one of:

        - ``'numeric'`` - for continuous ordinal outputs.
        - ``'categorical'`` - for discrete, non-ordinal outputs. These can still be numbers,
          but these numbers do not have inherent value.

        For more on image / label properties, see the guide about :ref:`vision_properties_guide`.
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
    >>> suite = train_test_validation()
    >>> train_data, test_data = ...
    >>> result = suite.run(train_data, test_data, max_samples=800)
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

    return Suite('Train Test Validation Suite', NewLabels(**kwargs).add_condition_new_label_ratio_less_or_equal(),
                 HeatmapComparison(**kwargs), TrainTestLabelDrift(**kwargs).add_condition_drift_score_less_than(),
                 ImagePropertyDrift(**kwargs).add_condition_drift_score_less_than(), ImageDatasetDrift(**kwargs),
                 PropertyLabelCorrelationChange(**kwargs).add_condition_property_pps_difference_less_than(), )


def model_evaluation(scorers: Union[Dict[str, Union[Callable, str]], List[Any]] = None,
                     area_range: Tuple[float, float] = (32 ** 2, 96 ** 2),
                     image_properties: List[Dict[str, Any]] = None, prediction_properties: List[Dict[str, Any]] = None,
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
           * - :ref:`plot_weak_segment_performance`
             - :class:`~deepchecks.vision.checks.model_evaluation.WeakSegmentPerformance`

    Parameters
    ----------
    scorers: Union[Dict[str, Union[Callable, str]], List[Any]], default: None
        Scorers to override the default scorers (metrics), find more about the supported formats at
        https://docs.deepchecks.com/stable/user-guide/general/metrics_guide.html
    area_range: tuple, default: (32**2, 96**2)
        Slices for small/medium/large buckets. (For object detection tasks only)
    image_properties : List[Dict[str, Any]], default: None
        List of properties. Replaces the default deepchecks properties.
        Each property is a dictionary with keys ``'name'`` (str), ``method`` (Callable) and ``'output_type'`` (str),
        representing attributes of said method. 'output_type' must be one of:

        - ``'numeric'`` - for continuous ordinal outputs.
        - ``'categorical'`` - for discrete, non-ordinal outputs. These can still be numbers,
          but these numbers do not have inherent value.

        For more on image / label properties, see the guide about :ref:`vision_properties_guide`.
    prediction_properties : List[Dict[str, Any]], default: None
        List of properties. Replaces the default deepchecks properties.
        Each property is a dictionary with keys ``'name'`` (str), ``method`` (Callable) and ``'output_type'`` (str),
        representing attributes of said method. 'output_type' must be one of:

        - ``'numeric'`` - for continuous ordinal outputs.
        - ``'categorical'`` - for discrete, non-ordinal outputs. These can still be numbers,
          but these numbers do not have inherent value.
        - ``'class_id'`` - for properties that return the class_id. This is used because these
          properties are later matched with the ``VisionData.label_map``, if one was given.

        For more on image / label properties, see the guide about :ref:`vision_properties_guide`.
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
    >>> test_vision_data = ...
    >>> result = suite.run(test_vision_data, max_samples=800)
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

    return Suite('Model Evaluation Suite',
                 ClassPerformance(**kwargs).add_condition_train_test_relative_degradation_less_than(),
                 MeanAveragePrecisionReport(**kwargs).add_condition_average_mean_average_precision_greater_than(),
                 MeanAverageRecallReport(**kwargs),
                 TrainTestPredictionDrift(**kwargs).add_condition_drift_score_less_than(),
                 SimpleModelComparison(**kwargs).add_condition_gain_greater_than(), ConfusionMatrixReport(**kwargs),
                 WeakSegmentsPerformance(**kwargs).add_condition_segments_relative_performance_greater_than(), )


def data_integrity(image_properties: List[Dict[str, Any]] = None, label_properties: List[Dict[str, Any]] = None,
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
        Each property is a dictionary with keys ``'name'`` (str), ``method`` (Callable) and ``'output_type'`` (str),
        representing attributes of said method. 'output_type' must be one of:

        - ``'numeric'`` - for continuous ordinal outputs.
        - ``'categorical'`` - for discrete, non-ordinal outputs. These can still be numbers,
          but these numbers do not have inherent value.

        For more on image / label properties, see the guide about :ref:`vision_properties_guide`.
    label_properties : List[Dict[str, Any]], default: None
        List of properties. Replaces the default deepchecks properties.
        Each property is a dictionary with keys ``'name'`` (str), ``method`` (Callable) and ``'output_type'`` (str),
        representing attributes of said method. 'output_type' must be one of:

        - ``'numeric'`` - for continuous ordinal outputs.
        - ``'categorical'`` - for discrete, non-ordinal outputs. These can still be numbers,
          but these numbers do not have inherent value.
        - ``'class_id'`` - for properties that return the class_id. This is used because these
          properties are later matched with the ``VisionData.label_map``, if one was given.

        For more on image / label properties, see the guide about :ref:`vision_properties_guide`.
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
    >>> vision_data = ...
    >>> result = suite.run(vision_data, max_samples=800)
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

    return Suite('Data Integrity Suite', ImagePropertyOutliers(**kwargs), LabelPropertyOutliers(**kwargs),
                 PropertyLabelCorrelation(**kwargs))


def full_suite(n_samples: Optional[int] = 5000, image_properties: List[Dict[str, Any]] = None,
               label_properties: List[Dict[str, Any]] = None, prediction_properties: List[Dict[str, Any]] = None,
               scorers: Union[Dict[str, Union[Callable, str]], List[Any]] = None,
               area_range: Tuple[float, float] = (32 ** 2, 96 ** 2), **kwargs) -> Suite:
    """Create a suite that includes many of the implemented checks, for a quick overview of your model and data.

    Parameters
    ----------
    n_samples : Optional[int] , default : 5000
        Number of samples to use for the checks in the suite. If None, all samples will be used.
    image_properties : List[Dict[str, Any]], default: None
        List of properties. Replaces the default deepchecks properties.
        Each property is a dictionary with keys ``'name'`` (str), ``method`` (Callable) and ``'output_type'`` (str),
        representing attributes of said method. 'output_type' must be one of:

        - ``'numeric'`` - for continuous ordinal outputs.
        - ``'categorical'`` - for discrete, non-ordinal outputs. These can still be numbers,
          but these numbers do not have inherent value.

        For more on image / label properties, see the guide about :ref:`vision_properties_guide`.
    label_properties : List[Dict[str, Any]], default: None
        List of properties. Replaces the default deepchecks properties.
        Each property is a dictionary with keys ``'name'`` (str), ``method`` (Callable) and ``'output_type'`` (str),
        representing attributes of said method. 'output_type' must be one of:

        - ``'numeric'`` - for continuous ordinal outputs.
        - ``'categorical'`` - for discrete, non-ordinal outputs. These can still be numbers,
          but these numbers do not have inherent value.
        - ``'class_id'`` - for properties that return the class_id. This is used because these
          properties are later matched with the ``VisionData.label_map``, if one was given.

        For more on image / label properties, see the guide about :ref:`vision_properties_guide`.

    scorers: Union[Dict[str, Union[Callable, str]], List[Any]], default: None
        Scorers to override the default scorers (metrics), find more about the supported formats at
        https://docs.deepchecks.com/stable/user-guide/general/metrics_guide.html
    area_range: tuple, default: (32**2, 96**2)
        Slices for small/medium/large buckets. (For object detection tasks only)
    image_properties : List[Dict[str, Any]], default: None
        List of properties. Replaces the default deepchecks properties.
        Each property is a dictionary with keys ``'name'`` (str), ``method`` (Callable) and ``'output_type'`` (str),
        representing attributes of said method. 'output_type' must be one of:

        - ``'numeric'`` - for continuous ordinal outputs.
        - ``'categorical'`` - for discrete, non-ordinal outputs. These can still be numbers,
          but these numbers do not have inherent value.

        For more on image / label properties, see the guide about :ref:`vision_properties_guide`.
    prediction_properties : List[Dict[str, Any]], default: None
        List of properties. Replaces the default deepchecks properties.
        Each property is a dictionary with keys ``'name'`` (str), ``method`` (Callable) and ``'output_type'`` (str),
        representing attributes of said method. 'output_type' must be one of:

        - ``'numeric'`` - for continuous ordinal outputs.
        - ``'categorical'`` - for discrete, non-ordinal outputs. These can still be numbers,
          but these numbers do not have inherent value.
        - ``'class_id'`` - for properties that return the class_id. This is used because these
          properties are later matched with the ``VisionData.label_map``, if one was given.

        For more on image / label properties, see the guide about :ref:`vision_properties_guide`.
    Returns
    -------
    Suite
        A suite that includes integrity checks.
    """
    args = locals()
    args.pop('kwargs')
    non_none_args = {k: v for k, v in args.items() if v is not None}
    kwargs = {**non_none_args, **kwargs}
    return Suite('Full Suite', model_evaluation(**kwargs), train_test_validation(**kwargs), data_integrity(**kwargs))
