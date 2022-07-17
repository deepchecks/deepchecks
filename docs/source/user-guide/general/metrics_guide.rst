.. _metrics_guide:

====================
Metrics Guide
====================

This guide will help you understand how to control the metrics that validate and monitor your data with
Deepchecks.
The assessment of how well the model performs should drive informed decision making regarding the usage of that model,
so we have to make sure that we measure the right things, i.e. the criteria that represnt tha key factors in the
decision making process.

**Structure:**

* `How can I use metrics with Deepchecks? <#how-can-i-use-metrics-with-deepchecks>`__
* `Default metrics <#default-metrics>`__
* `Alternative metrics <#alternative-metrics>`__



How can I use metrics with Deepchecks?
======================================
The metrics used in deepchecks are in accordance with sklearn scorer API for tabular checks and with Torch Ignite metric
API for vision checks.
To use metrics with Deepchecks suites and checks you can use the defaults, that fits the most common usecase, or pass
alternative scorers that are more accurate for your specific usecase.

Default metrics
===============
All of our relevant checks come with default metrics.

The default scorers by task type are:

Tabular:

Classification, binary or multiclass averaged over the classes:

*   Accuracy
*   Precision
*   Recall

Multiclass classification per class:

*   F1
*   Precision
*   Recall

Regression:

*   Negative RMSE (negative so larger is better)
*   Negative MAE
*   R2

Vision:

Classification:

*   Precision
*   Recall

Object detection:

*   Mean average precision
*   Mean average recall

To run a check with the default metrics, run it wihout passing the scorers.

:doc:`ClassPerformance </checks_gallery/vision/model_evaluation/plot_class_performance>`:

.. code-block:: python

    check = ClassPerformance()
    result = check.run(train_ds, test_ds, mnist_model)
    result


Alternative metrics
===================
Sometimes the defaults aren’t good enough to describe the specifics of the problem.
If this is the case, you can pass a list of metrics as a parameter to the check.

The metrics on the list can be existing
`Ignite metrics <https://pytorch.org/ignite/metrics.html#complete-list-of-metrics>`__ or
`Sklearn scorers <https://scikit-learn.org/stable/modules/model_evaluation.html>`__ or your own implementations.


Custom metrics
--------------
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
    suite = model_evaluation(alternative_scorers=custom_scorers)
    result = suite.run(train_dataset, test_dataset, model)


.. code-block:: python

    from ignite.metrics import Precision

    precision = Precision(average=True)
    double_precision = 2 * precision

    check = SingleDatasetPerformance({'precision2': double_precision})
    result = check.run(train_ds, mnist_model)
    result.value
