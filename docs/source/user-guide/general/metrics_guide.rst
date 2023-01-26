.. _metrics_guide:

====================
Metrics Guide
====================

In this guide we'll explain how to customize the metrics that deepchecks uses to validate and monitor your model
performance.
Controlling the metrics helps you shape the checks and suites according to the specifics of your use case.

**Structure:**

* `Default Metrics <#default-metrics>`__
* `Alternative Metrics <#alternative-metrics>`__  - How to pass to the checks your desired metrics and override the
  default ones.
* `List of Supported Strings <#list-of-supported-strings>`__ - List of the names of the pre-implemented metrics that can
  be passed to the checks.
* `Custom Metrics <#custom-metrics>`__ - How to pass to the checks a custom metric that is not in the pre-implemented
  ones.


Default Metrics
===============
All of the checks that evaluate model performance, such as
:doc:`SingleDatasetPerformance </checks_gallery/vision/model_evaluation/plot_single_dataset_performance>`
come with default metrics.

The default metrics by task type are:

Tabular
_______

Binary classification:

*   Accuracy ``'accuracy'``
*   Precision ``'precision'``
*   Recall ``'recall'``

Multiclass classification averaged over the classes:

*   Accuracy ``'accuracy'``
*   Precision ``'precision_macro'``
*   Recall ``'recall_macro'``

Multiclass classification per class:

*   F1 ``'f1_per_class'``
*   Precision ``'precision_per_class'``
*   Recall ``'recall_per_class'``

Regression:

*   Negative RMSE ``'neg_rmse'``
*   Negative MAE ``'neg_mae'``
*   R2 ``'r2'``

.. _metrics_guide_note_regression:
.. Note::
    Deepchecks follow the convention that greater metric value represent better performance.
    Therefore, it is recommended to only use metrics that follow
    this convention, for example, Negative MAE instead of MAE.

Vision
______

Classification:

*   Precision ``'precision_per_class'``
*   Recall ``'recall_per_class'``

Object detection:

*   Mean average precision ``'average_precision_per_class'``
*   Mean average recall  ``'average_recall_per_class'``

Running a Check with Default Metrics
____________________________________

To run a check with the default metrics, run it without passing any value to the "scorers"
parameter. We will demonstrate it using the
:doc:`ClassPerformance </checks_gallery/vision/model_evaluation/plot_class_performance>` check:

.. literalinclude:: ../../../../examples/examples_metrics_guide.py
    :language: python
    :lines: 26-31
    :tab-width: 0

Alternative Metrics
===================
Sometimes the defaults don't fit the specifics of the use case.
If this is the case, you can pass a list of supported metric strings or a dict in the format
{``metric_name_string``: ``metric``} to the scorers parameter of the check or suite.

The metrics in the dict can be some of the existing:

*   Strings from Deepchecks' `supported strings <#list-of-supported-strings>`__ for both vision and tabular.
*   `Scikit-learn Scorers <https://scikit-learn.org/stable/modules/model_evaluation.html>`__ for both vision and tabular.
    A Scikit-learn Scorer is a function that accepts the parameters: (model, x, y_true), and returns a score with the
    convention that higher is better. This is the method for developing custom tabular metrics, and is also the advised
    method for developing custom vision metrics for classification tasks.
*   :py:mod:`Deepchecks Metrics <deepchecks.vision.metrics>` for vision Metrics implemented by Deepchecks. These are
    dedicated metrics for object detection and semantic segmentation, such as the ``MeanIoU`` metric. Using them is
    advised when, for example, defining custom confidence or custom IoU thresholds is needed.
    You can import them from ``deepchecks.vision.metrics``.
*   For cases in which a new vision custom metrics is needed, such as for implementing additional object detection
    or segmentation metrics, deepchecks also supports custom metric classes.

Jump to the `Custom Metrics <#custom-metrics>`__ section for further information about implementing your own metrics
using the Scikit-learn Scorer api or a custom metric class.


Example for passing strings:

.. literalinclude:: ../../../../examples/examples_metrics_guide.py
    :language: python
    :lines: 2-9
    :tab-width: 0

Example for passing Deepchecks metrics:

.. literalinclude:: ../../../../examples/examples_metrics_guide.py
    :language: python
    :lines: 67-74
    :tab-width: 0


List of Supported Strings
=========================

In addition to the strings listed below, all Sklearn `scorer strings
<https://scikit-learn.org/stable/modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules>`__
apply for all tabular task types, and for computer vision classification tasks.

Regression
__________
.. list-table::
   :widths: 25 75 75
   :header-rows: 1

   * - String
     - Metric
     - Comments
   * - 'neg_rmse'
     - negative root mean squared error
     - higher value represents better performance
   * - 'neg_mae'
     - negative mean absolute error
     - higher value represents better performance
   * - 'rmse'
     - root mean squared error
     - not recommended, see :ref:`note <metrics_guide_note_regression>`.
   * - 'mae'
     - mean absolute error
     - not recommended, see :ref:`note <metrics_guide_note_regression>`.
   * - 'mse'
     - mean squared error
     - not recommended, see :ref:`note <metrics_guide_note_regression>`.
   * - 'r2'
     - R2 score
     -

Classification
______________
.. Note::
    For classification tasks, Deepchecks requires an ordered
    list of all possible classes (Can also be inferred from provided data and model).
    It is also recommended to supply the model's output probabilities per class, as they are required for some metrics and checks
    which will not work without them.
    See :doc:`link </user-guide/tabular/supported_models>` for additional information.
.. list-table::
   :widths: 25 75 75
   :header-rows: 1

   * - String
     - Metric
     - Comments
   * - 'accuracy'
     - classification accuracy
     - scikit-learn
   * - ‘roc_auc’
     - Area Under the Receiver Operating Characteristic Curve (ROC AUC) - binary
     - for multiclass averaging options see
       `scikit-learn's documentation <https://scikit-learn.org/stable/modules/model_evaluation.html>`__
   * - ‘roc_auc_per_class’
     - Area Under the Receiver Operating Characteristic Curve (ROC AUC) - score per class
     - for multiclass averaging options see
       `scikit-learn's documentation <https://scikit-learn.org/stable/modules/model_evaluation.html>`__
   * - 'f1'
     - F-1 - binary
     -
   * - 'f1_per_class'
     - F-1 per class - no averaging
     -
   * - 'f1_macro'
     - F-1 - macro averaging
     -
   * - 'f1_micro'
     - F-1 - micro averaging
     -
   * - 'f1_weighted'
     - F-1 - macro, weighted by support
     -
   * - 'precision'
     - precision
     - suffixes apply as with 'f1'
   * - 'recall' , 'sensitivity'
     - recall (sensitivity)
     - suffixes apply as with 'f1'
   * - 'fpr'
     - False Positive Rate - binary
     - suffixes apply as with 'f1'
   * - 'fnr'
     - False Negative Rate - binary
     - suffixes apply as with 'f1'
   * - 'tnr', 'specificity'
     - True Negative Rate - binary
     - suffixes apply as with 'f1'
   * - 'roc_auc'
     - AUC - binary
     -
   * - 'roc_auc_per_class'
     - AUC per class - no averaging
     -
   * - 'roc_auc_ovr'
     - AUC - One-vs-rest
     -
   * - 'roc_auc_ovo'
     - AUC - One-vs-One
     -
   * - 'roc_auc_ovr_weighted'
     - AUC - One-vs-rest, weighted by support
     -
   * - 'roc_auc_ovo_weighted'
     - AUC - One-vs-One, weighted by support
     -

Object Detection
________________
.. list-table::
   :widths: 25 75 75
   :header-rows: 1

   * - String
     - Metric
     - Comments
   * - 'average_precision_per_class'
     - average precision for object detection
     -
   * - 'average_precision_macro'
     - average precision macro averaging
     -
   * - 'average_precision_weighted'
     - average precision macro, weighted by support
     -
   * - 'average_recall_per_class'
     - average recall for object detection
     - suffixes apply as with 'average_precision'

.. _metrics_guide__custom_metrics:

Custom Metrics
==============
You can also pass your own custom metric to relevant checks and suites.

For tabular metrics and vision classification tasks the custom metrics function should follow the
`sklearn scorer <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html>`__ API, which is
a function that accepts the parameters: (model, x, y_true), and returns a score with the convention that higher is
better.

For other computer vision tasks, you should implement a Deepchecks CustomMetric. A Deepchecks CustomMetric is an object
that calculates a metric by accumulating information about the labels and predictions batch by batch, and then
finalizes the metric computation once all batches have been processed. The metric must
inherit from :class:`deepchecks.vision.metrics_utils.CustomMetric` and implement the following methods:
``reset``, ``update`` and ``compute``:
    * ``reset`` - Resets the metric to its initial state, resets any internal variables. Called by deepchecks before
      first call to the ``update`` method.
    * ``update`` - Called once for each batch in the data, this method updates the metric's internal state based on the
      labels and predictions of one batch. The method's signature should be ``update(self, output)``, where output is a
      tuple containing first ``y_pred`` which is the model's output and second ``y_true`` is the ground truth, both
      given  as lists of numpy objects, adhering to the
      :doc:`deepchecks format </user-guide/vision/supported_tasks_and_formats>`. For example, an
      object detection label would be a list where each element is a numpy array of bounding boxes annotations,
      and the prediction would be a list where each element is a numpy array of bounding boxes predictions, both in
      the :doc:`deepchecks format </user-guide/vision/supported_tasks_and_formats>`.
    * ``compute`` - Returns the metric's value based on the internal state. Can be either a single number, or a numpy
      array of containing a number for each class. This method is called only once, after all batches have been
      processed.

The ``update`` method is called on each batch of data, and the ``compute`` method is called to compute the final metric.

Note that in all cases, multiclass classification scorers should assume that the labels are given in a
`multi-label format <https://scikit-learn.org/stable/glossary.html#term-multilabel-indicator-matrices>`__ (a binary
matrix). Binary classification scorers should assume that the labels are given as 0 and 1.


Tabular Example
_______________

.. literalinclude:: ../../../../examples/examples_metrics_guide.py
    :language: python
    :lines: 12-23
    :tab-width: 0


Vision Example
______________

.. literalinclude:: ../../../../examples/examples_metrics_guide.py
    :language: python
    :lines: 34-64
    :tab-width: 0
