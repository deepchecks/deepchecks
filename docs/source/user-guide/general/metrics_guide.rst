.. _metrics_guide:

====================
Metrics Guide
====================

In this guide we'll explain how to customize the metrics that deepchecks uses to validate and monitor your model.
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
All of the checks that evaluate model performance, such as `SingleDatasetPerformance
</checks_gallery/vision/model_evaluation/plot_single_dataset_performance>`, come with default metrics.

The default metrics by task type are:

Tabular
_______

Binary classification:

*   Accuracy 'accuracy'
*   Precision 'precision'
*   Recall 'recall'

Multiclass classification averaged over the classes:

*   Accuracy 'accuracy'
*   Precision 'precision_macro'
*   Recall 'recall_macro'

Multiclass classification per class:

*   F1 'f1_per_class'
*   Precision 'precision_per_class'
*   Recall 'recall_per_class'

Regression:

*   Negative RMSE (negative RMSE so that the convention that greater is better is kept) 'neg_rmse'
*   Negative MAE 'neg_mae'
*   R2 'r2'

Vision
______

Classification:

*   Precision 'precision_per_class'
*   Recall 'recall_per_class'

Object detection:

*   Mean average precision 'average_precision_per_class'
*   Mean average recall  'average_recall_per_class'

To run a check with the default metrics, run it without passing any value to the "scorer" parameter. For example:

:doc:`ClassPerformance </checks_gallery/vision/model_evaluation/plot_class_performance>`:

.. literalinclude:: ../../../../examples/examples_metrics_guide.py
    :language: python
    :lines: 2-8
    :tab-width: 0

Alternative Metrics
===================
Sometimes the defaults don't fit the specifics of the use case.
If this is the case, you can pass a list of supported metric strings or a dict in the format {``metric_name_string``: ``metric``} as a
parameter to the check.

The metrics in the dict can be existing:

*   Strings from Deepchecks' `supported strings <#list-of-supported-strings>`__ for both vision and tabular.
*   `Ignite metrics <https://pytorch.org/ignite/metrics.html#complete-list-of-metrics>`__ for vision.
    An Ignite Metric is a class with the methods: reset, compute, and update, that iterates over batches of data and
    aggregates the result.
*   `Sklearn scorers <https://scikit-learn.org/stable/modules/model_evaluation.html>`__ for both vision and tabular.
    An Sklearn Scorer is a function that accepts the parameters: (model, x, y_true), and returns a score with the
    convention that higher is better.

or `your own implementation <#custom-metrics>`__.

:doc:`ClassPerformance </checks_gallery/vision/model_evaluation/plot_class_performance>`:

.. literalinclude:: ../../../../examples/examples_metrics_guide.py
    :language: python
    :lines: 11-18
    :tab-width: 0
List of Supported Strings
=========================

In addition to the strings listed below, all Sklearn `scorer strings
<https://scikit-learn.org/stable/modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules>`__
apply.

.. list-table:: Supported Strings
   :widths: 25 75 75
   :header-rows: 1

   * - String
     - Metric
     - Comments
   * - 'neg_rmse'
     - negative root mean squared error
     - Sklearn
   * - 'neg_mae'
     - negative mean absolute error
     - Sklearn
   * - 'f1_per_class'
     - F-1  per class - no averaging
     -
   * - 'fpr_per_class'
     - False Positive Rate per class - no averaging
     -
   * - 'fpr'
     - False Positive Rate - binary
     -
   * - 'fpr_macro'
     - False Positive Rate - macro averaging
     -
   * - 'fpr_micro'
     - False Positive Rate - micro averaging
     -
   * - 'fpr_weighted'
     - False Positive Rate - weighted macro averaging
     -
   * - 'fnr'
     - False Negative Rate - binary
     - suffixes apply as with 'fpr'
   * - 'tnr', 'specificity'
     - True Negative Rate - binary
     - suffixes apply as with 'fpr'
   * - 'precision'
     - precision
     - suffixes apply as with 'fpr'
   * - 'recall' , 'sensitivity'
     - recall (sensitivity)
     - suffixes apply as with 'fpr'
   * - 'average_precision_per_class'
     - average precision for object detection
     - only for object detection
   * - 'average_recall_per_class'
     - average recall for object detection
     - only for object detection


Custom Metrics
==============
You can also pass your own custom metric to relevant checks and suites.

Custom metrics should follow the
`Ignite Metric <https://pytorch.org/ignite/metrics.html#how-to-create-a-custom-metric>`__ API for computer vision or
`Sklearn scorer <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html>`__ API for tabular.

Tabular Example
_______________

.. literalinclude:: ../../../../examples/examples_metrics_guide.py
    :language: python
    :lines: 21-32
    :tab-width: 0


Vision Example
______________

.. literalinclude:: ../../../../examples/examples_metrics_guide.py
    :language: python
    :lines: 35-42
    :tab-width: 0
