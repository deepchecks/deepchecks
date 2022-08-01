.. _metrics_guide:

====================
Metrics Guide
====================

In this guide we'll explain how to customize the metrics that deepchecks uses to validate and monitor your model.
Controlling the metrics helps you shape the checks and suites according to the specifics of you use case.

**Structure:**

* `Default metrics <#default-metrics>`__
* `Alternative metrics <#alternative-metrics>`__
* `List of Supported Strings <#list-of-supported-strings>`__
* `Custom Metrics <#custom-metrics>`__


Default metrics
===============
All of the checks that evaluate model performance, such as SingleDatasetPerformance, come with
default metrics.

The default metrics by task type are:

Tabular
_______

Classification, binary or multiclass averaged over the classes:

*   Accuracy
*   Precision
*   Recall

Multiclass classification per class:

*   F1
*   Precision
*   Recall

Regression:

*   Negative RMSE (negative RMSE so that the convention that greater is better is kept)
*   Negative MAE
*   R2

Vision
______

Classification:

*   Precision
*   Recall

Object detection:

*   Mean average precision
*   Mean average recall

To run a check with the default metrics, run it without passing any value to the "scorer" parameter. For example:

:doc:`ClassPerformance </checks_gallery/vision/model_evaluation/plot_class_performance>`:

.. code-block:: python

    check = ClassPerformance()
    result = check.run(train_ds, test_ds, mnist_model)


Alternative metrics
===================
Sometimes the defaults don't fit the specifics of the use-case.
If this is the case, you can pass a list of supported metric strings or a dict in the format {metric name: metric} as a
parameter to the check.

The metrics in the dict can be existing:

*   `Ignite metrics <https://pytorch.org/ignite/metrics.html#complete-list-of-metrics>`__ for vision.
    An Ignite Metric is a class with the methods: reset, compute, and update, that iterates over batches of data and
    aggregates the result.
*   `Sklearn scorers <https://scikit-learn.org/stable/modules/model_evaluation.html>`__ for both vision and tabular.
    An Sklearn Scorer is a function that accepts the parameters: (model, x, y_true), and returns a score with the
    convention that higher is better.
*   strings from Deepchecks' `supported strings <#list-of-supported-strings>`__ for both vision and tabular.

or `your own implementation <#custom-metrics>`__.

.. code-block:: python

   train_ds, test_ds = adult.load_data(data_format='Dataset', as_train_test=True)
   model = adult.load_fitted_model()

   scorer = ['precision_per_class', 'recall_per_class', 'fnr_macro']
   check = TrainTestPerformance(scorers=scorer)
   result = check.run(train_ds, test_ds, model)
   result


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


Custom metrics
==============
You can also pass your own custom metric to relevant checks and suites.

Custom metrics should follow the
`Ignite Metric <https://pytorch.org/ignite/metrics.html#how-to-create-a-custom-metric>`__ API for computer vision or
`Sklearn scorer <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html>`__ API for tabular.

.. code-block:: python

    from deepchecks.tabular.datasets.classification import adult
    from deepchecks.tabular.suites import model_evaluation
    from sklearn.metrics import cohen_kappa_score, fbeta_score, make_scorer

    f1_scorer = make_scorer(fbeta_score, labels=[0, 1], average=None, beta=0.2)
    ck_scorer = make_scorer(cohen_kappa_score)
    custom_scorers = {'f1': f1_scorer, 'cohen': ck_scorer}

    train_ds, test_ds = adult.load_data(data_format='Dataset', as_train_test=True)
    model = adult.load_fitted_model()
    suite = model_evaluation(scorers=custom_scorers)
    result = suite.run(train_dataset, test_dataset, model)


.. code-block:: python

    from ignite.metrics import Precision

    precision = Precision(average=True)
    double_precision = 2 * precision

    check = SingleDatasetPerformance({'precision2': double_precision})
    result = check.run(train_ds, mnist_model)


