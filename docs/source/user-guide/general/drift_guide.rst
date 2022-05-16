.. _drift_user_guide:

====================
Drift User Guide
====================

(Is this needed if there's a menu on the right?)

**Structure:**

* `What Is Distribution Drift? <#what-is-distribution-drift>`__
* `Which Types of Drift Are There? <#which-types-of-drift-are-there>`__
* `What Are the Causes of Drift? <#what-are-the-causes-of-drift>`__
* `How Do You Detect Drift? <#how-do-you-detect-drift>`__
* `How Can I Use Deepchecks to Detect Drift? <#how-can-i-use-deepchecks-to-detect-drift>`__
* `Code Examples <#code-examples>`__


What Is Distribution Drift?
==========================

Distribution drift is a change in the underlying distribution of the data, the label or of the relationship between them. Drift happens over time,
and can occur either gradually or suddenly.

Common reasons of drift can be natural changes in the data, such as shopping habits changing for an online retailer
(for example, during the covid pandemic, virtual shopping has risen significantly and buyer patterns have changed), or
data acquisition and processing issues, such as a camera settings changing accidentally to have lower exposure.

So Why Is Drift So Important?
-----------------------------

In machine learning, our models are built to predict only on data they have seen before.
There's a misconception that machine learning models are some magical intelligent models, that can understand
any data presented to them. But the truth is that machine learning models learn to predict only on similar data
(or, more formally, data drawn from the same distribution the model was trained on).
If that data has changed, and the relationships between the different variables and target label have changed,
our model may not be as accurate as it was before.

Detecting drift is an important warning sign that our model may be not as accurate on the new data compared to the training data, and that it should be
adjusted or retrained on different data.
In production environments, detecting drift (and other measured derived from drift, such as model confidence) is often the only way to know that our model performance is deteriorating,
as in many cases the label is unknown for some time after the prediction is made.

It is important to note that not all changes in data represent drift. For example, periodic changes in data due to daily, weekly or
seasonal changes are usually not considered drift, as they can be predicted in advance and the model can learn from
past data samples.


Which Types of Drift Are There?
================================

In machine learning, we usually refer to 3 types of drift:

Concept Drift
-------------
Concept drift happens when the underline relation between the data and the label has changed. In layman terms, this means that our
data has changed in a way that affects our label, meaning a model trained on previous data will have difficulty predicting
on the new data, as the label can't be predicted in the same way anymore.

Concept drift will almost always require some changes to the model, and will require retraining it on newer data.

Data Drift
----------
Data drift is any change in the distribution of the data. It usually refers to changes that do not significantly affect the label.
However, even when not affecting the model, data drift is important, as it can expose other undetected issues.

Moreover, when labels are not available (as happens in many cases), data drift cannot be discerned from concept drift.

Label Drift
-----------
Label drift is the change in the distribution of the label itself. Note that this can also be caused by concept drift,
but here we discuss a change that means that the label can still be accurately predicted from the data, but its
distribution has changed.

(Is this interesting? Should I add this to each drift? Should I detail the causes more?)

Label drift is important for 2 reasons:

1. Your model can be improved - Machine learning models use the label's distribution when learning to predict it.
For example, if a dataset has 90% of label A and 10% of label B, the model may learn that label A is more common and
is a better prediction in cases where the label can't be inferred easily. Changes to this distribution can prompt us
to retrain the model on the new distribution, in order to more accurately simulate the current reality.
Note that this is not relevant when the training dataset is sampled so labels are evenly distributed.

2. You may need to take some action - Not all changes in data means we need to retrain our model.
However, we may need to act differently on it.
For example, a fraud prevention company predicts fraudulent payments in 80% accuracy. Suddenly, the company notices that
that they are getting more fraudulent payments recently. They are still predicted in 80% accuracy, but those 20% they
get wrong are now more costly, and therefore the company needs to adjust its actions - perhaps adjusting the model
decision threshold, or adding a manual review process.

For more on the different types of drift, `see here <https://deepchecks.com/data-drift-vs-concept-drift-what-are-the-main-differences/>`_


What Are the Causes of Drift?
==============================

(Is this interesting on its own? Or should this be a part of the description of each drift?)


How Do You Detect Drift?
=========================

In general, drift detection is done by comparing the newer and older data, and seeing whether they are derived from
the same underlying distribution. This is actually not a "yes or no" question - Instead, we ask "what is the probability
that these 2 distributions are the same?".

There are many methods to detect drift. Here, we will elaborate on 2:

Detection by Statistical Test
-----------------------------
This is the simplest and most common drift detection method.
This is done by sampling a single variable (that can either be a data feature, the label or even the prediction) and
running a statistical test that aims to measure the difference the 2 samples.
There are many tests that can do this, such as the Kolmogorov-Smirnov test, Jensen-Shannon Divergence and more.
In deepchecks, we found that the best results are given by:

* For continuous numeric distributions - `Wasserstein metric (Earth Movers Distance) <https://en.wikipedia.org/wiki/Wasserstein_metric>`__
* For discrete or categorical distributions - `Population Stability Index (PSI) <https://www.lexjansen.com/wuss/2017/47_Final_Paper_PDF.pdf>`__ or `Cramer's V <https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V>`__

These methods have the advantage of being simple to use and produce explainable results. However, they are limited by
checking each feature one at a time, and cannot detect drift that occurs over multiple variables (or they detect it
multiple times)

Detection by Domain Classifier
------------------------------

This is a method to detect multivariate drift, meaning that it can run on several variables, and even on the whole dataset.
This is done by training a model to classify if a sample came from the train dataset or the new (test or production) dataset.
If the classifier can easily predict which sample is from which dataset, it would mean that there are significant differences between these datasets.

The main advantage of this method is that it can also uncover covariate drift, meaning drift in the data that does not
affect the distribution of each individual variable, but does affect the relationship between them. For example, ?


How Can I Use Deepchecks to Detect Drift?
=========================================
Deepchecks can test your data for both concept drift and label drift, by using a variety of methods.

Tabular Data
------------

To detect `data <#data-drift>`__ or `concept drift <#concept-drift>`__, deepchecks offers the
:doc:`Feature Drift check </checks_gallery/tabular/distribution/plot_train_test_feature_drift>` which uses univariate
`statistical test <#detection-by-statistical-test>`__ and the :doc:`Whole Dataset Drift check</checks_gallery/tabular/distribution/plot_whole_dataset_drift>`
which uses a `domain classifier <#detection-by-domain-classifier>`__ in order to detect multivariate drift.

For label drift, deepchecks offers the :doc:`Label Drift check </checks_gallery/tabular/distribution/plot_train_test_label_drift>`, which also uses univariate `statistical test <#detection-by-statistical-test>`__.

In cases where the label is not available, we strongly recommend to also use the :doc:`Prediction Drift check</checks_gallery/tabular/distribution/plot_train_test_prediction_drift>`,
which uses the same methods but on the model's predictions, and can detect possible changes in the distribution of the label.

For code examples, see `here <#tabular-checks>`__

Computer Vision Data
--------------------

To detect `data <#data-drift>`__ or `concept drift <#concept-drift>`__, deepchecks offers the
:doc:`Image Property Drift check </checks_gallery/vision/distribution/plot_image_property_drift>` which uses univariate
`statistical test <#detection-by-statistical-test>`__ and the :doc:`Image Dataset Drift check</checks_gallery/vision/distribution/plot_image_dataset_drift>`
which uses a `domain classifier <#detection-by-domain-classifier>`__ in order to detect multivariate drift.

For label drift, deepchecks offers the :doc:`Label Drift check </checks_gallery/vision/distribution/plot_train_test_label_drift>`, which also uses univariate `statistical test <#detection-by-statistical-test>`__.

In cases where the label is not available, we strongly recommend to also use the :doc:`Prediction Drift check</checks_gallery/vision/distribution/plot_train_test_prediction_drift>`,
which uses the same methods but on the model's predictions, and can detect possible changes in the distribution of the label.

All of the computer vision checks use the :doc:`image and label properties</user-guide/vision-properties>` to estimate
drift, as image data and labels are not simple one-dimensional variables.

For code examples, see `here <#computer-vision-checks>`__


Code Examples
=============


Tabular Checks
--------------
TBC


Computer Vision Checks
----------------------

TBC