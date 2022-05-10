.. _drift_user_guide:

====================
Drift User Guide
====================

**Structure:**

* `What is a feature drift? <#what-is-a-feature-drift>`__
* `Generate data & model <#generate-data-model>`__
* `Run the check <#run-the-check>`__
* `Define a condition <#define-a-condition>`__

What is drift?

Which types of drift are there?

What are the causes of drift?

Which drift detection methods exist?

How can I use deepchecks to detect drift?


What Is Distribution Drift?
==============

Distribution drift is a change in the underlying distribution of the data, the target, or the prediction.

Drift happens over time, and can occur either slowly or suddenly.


Not all changes in data are drift. For example, periodic changes in data due to daily, weekly or seasonal changes are not considered drift.

Some changes in data are predictable - sales  As drift is a change in the distribution of data, it is inherently unpredictable. , and is not predictable like other changes in data, such as periodic
changes in data (e.g. daily or seasonal changes).








What is a feature drift?
========================
Data drift is simply a change in the distribution of data over time. It is
also one of the top reasons of a machine learning model performance degrades
over time.

Causes of data drift include:

* Upstream process changes, such as a sensor being replaced that changes the
  units of measurement from inches to centimeters.
* Data quality issues, such as a broken sensor always reading 0.
* Natural drift in the data, such as mean temperature changing with the seasons.
* Change in relation between features, or covariate shift.

Feature drift is such drift in a single feature in the dataset.

In the context of machine learning, drift between the training set and the
test set will likely make the model to be prone to errors. In other words,
this means that the model was trained on data that is different from the
current test data, thus it will probably make more mistakes predicting the
target variable.

How deepchecks detects feature drift
------------------------------------
There are many methods to detect feature drift. Some of them include
training a classifier that detects which samples come from a known
distribution and defines the drift by the accuracy of this classifier. For
more information, refer to the :doc:`Whole Dataset Drift check
</checks_gallery/tabular/distribution/plot_whole_dataset_drift>`.

Other approaches include statistical methods aim to measure difference
between distribution of 2 given sets. We exprimented with various approaches
and found that for detecting drift in a single feature, the following 2
methods give the best results:

* `Population Stability Index (PSI) <https://www.lexjansen.com/wuss/2017/47_Final_Paper_PDF.pdf>`__
* `Wasserstein metric (Earth Movers Distance) <https://en.wikipedia.org/wiki/Wasserstein_metric>`__

For numerical features, the check uses the Earth Movers Distance method
and for the categorical features it uses the PSI. The check calculates drift
between train dataset and test dataset per feature, using these 2 statistical
measures.


What Is Prediction Drift?
===========================
The term drift (and all it's derivatives) is used to describe any change in the data compared
to the data the model was trained on. Prediction drift refers to the case in which a change
in the data (data/feature drift) has happened and as a result, the distribution of the
models' prediction has changed.

Calculating prediction drift is especially useful in cases
in which labels are not available for the test dataset, and so a drift in the predictions
is our only indication that a changed has happened in the data that actually affects model
predictions. If labels are available, it's also recommended to run the `Label Drift Check
</examples/tabular/checks/distribution/examples/plot_train_test_label_drift.html>`__.

There are two main causes for prediction drift:

* A change in the sample population. In this case, the underline phenomenon we're trying
  to predict behaves the same, but we're not getting the same types of samples. For example,
  Iris Virginica stops growing and is not being predicted by the model trained to classify Iris species.
* Concept drift, which means that the underline relation between the data and
  the label has changed.
  For example, we're trying to predict income based on food spending, but ongoing inflation effect prices.
  It's important to note that concept drift won't necessarily result in prediction drift, unless it affects features that
  are of high importance to the model.

How Does the TrainTestPredictionDrift Check Work?
=================================================
There are many methods to detect drift, that usually include statistical methods
that aim to measure difference between 2 distributions.
We experimented with various approaches and found that for detecting drift between 2
one-dimensional distributions, the following 2 methods give the best results:

* For regression problems, the `Population Stability Index (PSI) <https://www.lexjansen.com/wuss/2017/47_Final_Paper_PDF.pdf>`__
* For classification problems, the `Wasserstein Distance (Earth Mover's Distance) <https://en.wikipedia.org/wiki/Wasserstein_metric>`__



What is a dataset drift?
========================
A whole dataset drift, or a multivariate dataset drift, occurs when the
statistical properties of our input feature change, denoted by a change
in the distribution P(X).

Causes of data drift include:

* Upstream process changes, such as a sensor being replaced that changes
  the units of measurement from inches to centimeters.
* Data quality issues, such as a broken sensor always reading 0.
* Natural drift in the data, such as mean temperature changing with the seasons.
* Change in relation between features, or covariate shift.

The difference between a :doc:`feature drift
</checks_gallery/tabular/distribution/plot_train_test_feature_drift>`
(or univariate dataset drift) and a multivariate drift is that in the
latter the data drift occures in more that one feature.

In the context of machine learning, drift between the training set and the
test means that the model was trained on data that is different from the
current test data, thus it will probably make more mistakes predicting the
target variable.

How deepchecks detects dataset drift
------------------------------------
There are many methods to detect feature drift. Some of them are statistical
methods that aim to measure difference between distribution of 2 given sets.
This methods are more suited to univariate distributions and are primarily
used to detect drift between 2 subsets of a single feature.

Measuring a multivariate data drift is a bit more challenging. In the whole
dataset drift check, the multivariate drift is measured by training a classifier
that detects which samples come from a known distribution and defines the
drift by the accuracy of this classifier.

Practically, the check concatanates the train and the test sets, and assigns
label 0 to samples that come from the training set, and 1 to those who are
from the test set. Then, we train a binary classifer of type
`Histogram-based Gradient Boosting Classification Tree
<https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html>`__, and measure the
drift score from the AUC score of this classifier.


What Is Dataset Drift?
------------------------
Data drift is simply a change in the distribution of data over time. It is also
one of the top reasons that a machine learning model performance degrades over time.

Specifically, a whole dataset drift, or a multivariate dataset drift, occurs when
there is a change in the relation between input features.

Causes of data drift include:

* Natural drift in the data, such as lighting (brightness) changes between summer
  and winter.
* Upstream process changes, such as a camera being replaced that has a different
  lens, which makes images sharper.
* Data quality issues, such as a malfunctioning camera that always returns a black image.
* Data pipeline errors, such as a change in image augmentations done in preprocessing.

In the context of machine learning, drift between the training set and the test set
(which is not due to augmentation) will likely make the model prone to errors. In
other words, if the model was trained on data that is different from the current test
data, it will probably make more mistakes predicting the target variable.

How Does the ImageDatasetDrift Check Work?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
There are many methods to detect feature drift. Some of them are statistical methods
that aim to measure difference between distribution of 2 given sets. This methods
are more suited to univariate distributions and are primarily used to detect drift
between 2 subsets of a single feature.

Measuring a multivariate data drift is a bit more challenging. In the image dataset
drift check, the multivariate drift is measured by training a classifier that detects
which samples come from a known distribution and defines the drift by the accuracy
of this classifier.

Practically, the check concatenates the train and the test sets, and assigns label 0
to samples that come from the training set, and 1 to those from the test set.
Then, we train a binary classifer of type `Histogram-based Gradient Boosting
Classification Tree <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html>`_,
and measure the drift score from the AUC score of this classifier.

As the classifier is a tree model, that cannot run on the images themselves, the
check calculates properties for each image (such as brightness, aspect ratio etc.)
and uses them as input features to the classifier.

Which Image Properties Are Used?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
==============================  ==========
Property name                   What is it
==============================  ==========
Aspect Ratio                    Ratio between height and width of image (height / width)
Area                            Area of image in pixels (height * width)
Brightness                      Average intensity of image pixels. Color channels have different weights according to
                                RGB-to-Grayscale formula
RMS Contrast                    Contrast of image, calculated by standard deviation of pixels
Mean Red Relative Intensity     Mean over all pixels of the red channel, scaled to their relative intensity in
                                comparison to the other channels [r / (r + g + b)].
Mean Green Relative Intensity   Mean over all pixels of the green channel, scaled to their relative intensity in
                                comparison to the other channels [g / (r + g + b)].
Mean Blue Relative Intensity    Mean over all pixels of the blue channel, scaled to their relative intensity in
                                comparison to the other channels [b / (r + g + b)].
==============================  ==========
"""
