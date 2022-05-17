.. _drift_user_guide:

====================
Drift User Guide
====================

**In This Guide, You Can Learn:**

* `What Is Distribution Drift and How Can You Detect It? <#what-is-distribution-drift>`__
* `How Can You Use Deepchecks to Detect Drift? <#how-can-i-use-deepchecks-to-detect-drift>`__


What Is Distribution Drift?
==========================

Distribution drift is a change in the underlying distribution of the data, the label or of the relationship between them. Drift happens over time,
and can occur either gradually or suddenly.

Common reasons of drift can be natural changes in the data, such as shopping habits changing for an online retailer
(for example, during the covid pandemic, virtual shopping has risen significantly and buyer patterns have changed), or
data acquisition and processing issues, such as a camera settings changing accidentally to have lower exposure.

So Why Is Drift So Important?
-----------------------------

Machine learning models are meant to predict on unseen data, based on previous known data (more accurately, this refers
to supervised machine learning models, for which drift is relevant).
If the data, or the relationships between the different variables and target label, has changed, our model may not be as
accurate as it was before.

Detecting drift is an important warning sign that our model may be not as accurate on the new data compared to the training data, and that it should be
adjusted or retrained on different data.
In production environments, detecting drift (and other measured derived from drift, such as model confidence) is often the only way to know that our model performance is deteriorating,
as in many cases the label is unknown for some time after the prediction is made.

It is important to note that not all changes in data represent drift. For example, periodic changes in data due to daily, weekly or
seasonal changes are usually not considered drift, as they as they are often present in the training data.


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

In many cases, label drift alone might not be of interest. However, it may affect your model's accuracy, as less-accurate
classes may be more prevalent now that they were in your train dataset.

For more on the different types of drift, `see here <https://deepchecks.com/data-drift-vs-concept-drift-what-are-the-main-differences/>`_

What Can You Do in Case of Drift?
---------------------------------

There are several solutions to distribution drift. They are also dependent on the information you have.

* If your drifted data has labels, you can easily know which type of drift you are facing.
* However, the more common case is when you don't have your data labels yet. In this case, we can only try and measure
  drift on the data (features) or the model's output.
  If we detect drift, we have no way to know whether this drift is a simple data drift, label drift or concept drift, as
  we don't know the true labels.

(Feels disconnected to the rest here)

Retrain Your Model
^^^^^^^^^^^^^^^^^^

If you have either kind of drift, retraining your model on new data, that better represents the current distribution
of data, is the most straight-forward solution.
However, this solution may require additional resources such as manual labeling of new data, or might not be possible
if labels on the newer data are not available yet.

Retraining is usually necessary in cases of concept drift. However, retraining may still be of use even for label drift,
as the model may perform better when knowing the correct distribution of the label (this is not relevant when the
training dataset is sampled so labels are evenly distributed)


Adjust Your Prediction
^^^^^^^^^^^^^^^^^^^^^^

When retraining is not an option, or if a quick action needs to be taken, adjustments to the output of the models may
still help in cases of concept drift. This can be done by either recalibrating your model's output, or by changing your
decision thresholds on the model's scores.

However, these methods assume that there's still enough similarity between your training data and your current data,
which may not always be the case.

Do Nothing
^^^^^^^^^^

Not all drift is necessarily bad, and each case should be examined separately. Sometimes, data drift may be simply
explained by label drift (for example, data drift of social media pictures  can be explained by the label drift of ... ) ??


How Do You Detect Drift?
=========================

In general, drift detection is done by comparing the newer and older data, and seeing whether they are derived from
the same underlying distribution. This is actually not a "yes or no" question - Instead, we ask "what is the probability
that these 2 distributions are the same?".

There are many methods to detect drift. Here, we will elaborate on 2:

Detection by Univariate Measure
--------------------------------
This is the simplest and most common drift detection method.
This is done by taking only one variable at a time (that can either be a data feature, the label or even the prediction)
and measuring the difference between newer and older samples of the variable.
There are many measures that can be used for this, such as the Kolmogorov-Smirnov test, Jensen-Shannon Divergence and more.
In deepchecks, we found that the best results are given by:

* For continuous numeric distributions - `Wasserstein metric (Earth Movers Distance) <https://en.wikipedia.org/wiki/Wasserstein_metric>`__
* For discrete or categorical distributions - `Population Stability Index (PSI) <https://www.lexjansen.com/wuss/2017/47_Final_Paper_PDF.pdf>`__ or `Cramer's V <https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V>`__

These methods have the advantage of being simple to use and produce explainable results. However, they are limited by
checking each feature one at a time, and cannot detect drift in the relations between features. Also, these methods
will usually detect drift multiple times if it occures in several features.

Detection by Domain Classifier
------------------------------

This is a method to detect multivariate drift, meaning that it can run on several variables, and even on the whole dataset.
This is done by training a model to classify if a sample came from the train dataset or the new (test or production) dataset.
If the classifier can easily predict which sample is from which dataset, it would mean that there are significant differences between these datasets.

The main advantage of this method is that it can also uncover covariate drift, meaning drift in the data that does not
affect the distribution of each individual variable, but does affect the relationship between them.

For example, you're predicting the income of a person from his city and education. Let's say a tech giant now moved into city A. This means that:
1. Given that a person lives in city A, he's more likely to have a more advanced degree (educated people moved to city A) - this is multivariate drift.
2. Given his education, a resident of city A now earns more. - this is concept drift.


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

All of these checks appear in the `deepchecks interactive demo <https://checks-demo.deepchecks.com>`__, where you can
insert corruption into the data and see the check at work.

Computer Vision Data
--------------------

All of the computer vision checks use the :doc:`image and label properties</user-guide/vision-properties>` to estimate
drift, as image data and labels are not simple one-dimensional variables.

To detect `data <#data-drift>`__ or `concept drift <#concept-drift>`__, deepchecks offers the
:doc:`Image Property Drift check </checks_gallery/vision/distribution/plot_image_property_drift>` which uses univariate
`statistical test <#detection-by-statistical-test>`__ and the :doc:`Image Dataset Drift check</checks_gallery/vision/distribution/plot_image_dataset_drift>`
which uses a `domain classifier <#detection-by-domain-classifier>`__ in order to detect multivariate drift.

For label drift, deepchecks offers the :doc:`Label Drift check </checks_gallery/vision/distribution/plot_train_test_label_drift>`, which also uses univariate `statistical test <#detection-by-statistical-test>`__.

In cases where the label is not available, we strongly recommend to also use the :doc:`Prediction Drift check</checks_gallery/vision/distribution/plot_train_test_prediction_drift>`,
which uses the same methods but on the model's predictions, and can detect possible changes in the distribution of the label.

For code examples, see `here <#computer-vision-checks>`__


Code Examples
=============


Tabular Checks
--------------
TBC


Computer Vision Checks
----------------------

TBC