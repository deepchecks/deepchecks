.. _metrics_guide:

====================
Metrics Guide
====================

This guide will help you understand what metrics are and how you can use them to validate and monitor your data with
Deepchecks.

**Structure:**

* `Why do we need metrics? <#why_do_we_need_metrics>`__
* `What is a metric? <#what_is_a_metric?>`__
* `How are metrics different from loss functions? <#metrics_vs_loss_functions>`__
* `Common metrics <#common_metrics>`__
* `How can I use metrics with Deepchecks? <#using_metrics_with_deepchecks>`__
* `Code examples <#code_examples>`__

Why do we need metrics?
=======================
When asking how well a model performs, we want to get an answer that’s more informative than “nice”.
The assessment of how well the model performs should drive informed decision making regarding the usage of that model,
and for that “nice” is just not enough.
Some usages for such assessment are:

* Defining a goal for the model
* Comparing different models
* Monitoring performance changes
* Communicating all of the above to colleagues, clients, etc.

What is a good metric?
----------------------
There is no metric for the goodness of metrics, but for the metric to be useful the following four characteristics
should generally apply:

1.  Deterministic - for the metric to be reliable it has to be reproducible, meaning that given the same data and the
    same model, the metric will always return the same result. It should be precisely defined. For that matter, a metric cannot
    rely on personal taste and opinions as those change between viewers and are therefore not reproducible.
2.  Comparable - for the metric result to be meaningful, we need to be able to compare it to other results.
    For that, the result has to be ordinal, so we can answer questions such as: “is the current result better or worse
    than the previous result?”. Moreover, the result should be on a clear scale so we would also be able to answer
    questions such as: “how much better is the current result compared to the previous?”
3.  Explainable- we would like to communicate the results to other stakeholders, as well as getting some intuition
    regarding the meaning of the results.
4.  Resonates with domain/business sense - this point is too often overlooked. We have to make sure that we are
    measuring the right thing, that is the level of success in the actual problem that the model is meant to solve.
    For example, if we use object detection for inventory count of a certain product, we care more about how many
    targets were detected rather than the targets’ exact location, so the fraction of predictions above a certain IoU
    would be more relevant than the IoU itself.

.. image:: /_static/images/general/tomato_sauce_on_shelf.png
   :alt: inventory object detection example
   :align: center
   :width: 400

Using object detection for inventory count illustration by Sol Yarkoni


Other consideration for choosing a metric might be:

* Time to implement- we might prefer a metric that can be used off the self.
* Using an academy or industry standard.
* Computation time and memory.

How are metrics different from loss functions?
==============================================
Metrics are used for model *evaluation* and loss functions are used for model *optimization*.

Since loss functions are used as objectives in optimization problems they need to satisfy some conditions that metrics
don’t such as differentiability.

During the optimization process the loss function is called many times, while the metric function is only called once at
the end of the process (except in some specific cases like early stopping <link> condition) so the calculation speed is
more crucial for loss functions than for metrics.

On the other hand, loss functions face inward- the results are returned to the optimizer (the entity managing the
optimization process), while metrics face outward, presenting the results to the user and because of that explainability
is more crucial for metrics than loss functions.

As a rule of thumb, loss function should be minimized and therefore a smaller result means better performance, while for
most metrics larger means better.

Many functions can be used as both metric or loss function. In this case an adjustment for the direction (minimize loss
or maximize metric) like a minus sign is commonly used.

Common Metrics
==============
Regression
----------
*   MSE - mean squared error, the average squared difference between the estimated values and the actual values.
    Probably the most common metric for regression.
*   MAE - mean absolute error, the average absolute difference between the estimated values and the actual values.
*   Cosine similarity - the cosine of the angle between two sequences or vectors of numbers. The cosine similarity does
    not depend on the magnitudes of the vectors, only on their angle, similar vectors are vectors pointing in the same
    direction.

Classification
--------------
In general, the common classification metrics are based on the concept of comparing the predicted label with the ground
truth label and counting the matches and mismatches between them.

This is usually done with the assistance of a confusion matrix. If you are not familiar with the concept of confusion
matrix, check out this link.

.. image:: /_static/images/general/truth_table.png
   :alt: Truth Table
   :align: center

Truth table by Sol Yarkoni

*   Accuracy - how many samples were classified correctly out of the total number of samples. Though very intuitive, it
    can often be misleading, especially for imbalanced data.
*   Precision- how many samples were classified correctly as positive out of the total number of samples classified as
    positive. Can be also viewed as the fraction of relevant samples out of the samples spotted by the model.
*   Recall/ Sensitivity/ TPR - how many samples were classified correctly as positive out of the total number of
    positive samples. Can be also viewed as the fraction of the relevant samples that were spotted by the model.


.. image:: /_static/images/general/PrecisionRecall_wikipedia.svg.png
   :alt: Precision and recall
   :align: center

Precision and Recall by Walber - Own work, CC BY-SA 4.0,
`source <https://commons.wikimedia.org/w/index.php?curid=36926283>`__


*   Specificity/ TNR - how many samples were classified correctly as negative out of the total number of negative
    samples. Complementary to sensitivity.

.. image:: /_static/images/general/Sensitivity_and_specificity_wikipedia.svg.png
   :alt: Specificity and sensitivity
   :align: center

Specificity and Sensitivity by FeanDoe - Modified version from Walber&#039;s Precision and Recall
`source <https://commons.wikimedia.org/w/index.php?curid=94134880>`__

*   F-1 - combines the precision and recall into one metric by taking their harmonic mean. More robust to class
    imbalance than accuracy.
*   AUC - The area under the curve of the ROC graph. For the previous classification metrics, a threshold on the model
    output was chosen above which the sample is classified as positive and below as negative. The AUC takes into account
    all possible thresholds.

Object Detection
----------------
*   IoU/ Jaccard Index - the ratio between the overlapping area of predicted and the actual bounding box and the union
    of their areas. The most intuitive metric for object detection.
*   mAP - mean average precision. The mean of the average precision per class over the classes. Calculated at a certain
    threshold, usually 0.5. Commonly used for benchmarking object detection models. A good explanation of how it is
    calculated can be found here <link>.
*   mAR - mean average recall. The mean of the average precision per class over the classes. Summed over the threshold
    range [0.5, 1].

How can I use metrics with Deepchecks?
======================================
The metrics used in deepchecks are in accordance with sklearn scorer API for tabular checks and with Torch Ignite metric
API for vision checks.

Default metrics
---------------
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

Alternative metrics
-------------------
As mentioned in the section “what is a good metric” the metrics should resonate with the logic of the problem at hand,
and that means that sometimes the defaults aren’t good enough to describe the specifics of the problem.

If this is the case, you can pass a list of metrics as a parameter to the check.

The metrics on the list can be existing Ignite metrics or Sklearn scorers or your own implementations.

Custom metrics
--------------
You can also pass your own custom metric to relevant checks and suites.

Custom metrics should follow the Ignite Metric API for computer vision or Sklearn scorer API for tabular.

