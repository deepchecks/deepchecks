# -*- coding: utf-8 -*-
"""
.. _nlp__prediction_drift:

================
Prediction Drift
================

This notebook provides an overview for using and understanding the NLP prediction drift check.

**Structure:**

* `What Is Prediction Drift? <#what-is-prediction-drift>`__
* `Get Data and Predictions <#get-data-and-predictions>`__
* `Run Check <#run-check>`__

What Is Prediction Drift?
=========================

Drift is simply a change in the distribution of data over time, and it is
also one of the top reasons why machine learning model's performance degrades
over time.

Prediction drift is when drift occurs in the prediction itself.
Calculating prediction drift is especially useful in cases
in which labels are not available for the test dataset, and so a drift in the predictions
is our only indication that a changed has happened in the data that actually affects model
predictions. If labels are available, it's also recommended to run the
:ref:`Label Drift check <nlp__label_drift>`.

For more information on drift, please visit our :ref:`drift guide <drift_user_guide>`.

How Deepchecks Detects Prediction Drift
---------------------------------------

This check detects prediction drift by using :ref:`univariate measures <drift_detection_by_univariate_measure>`
on the prediction output.
"""


#%%
# Get Data and Predictions
# ========================
#
# For this example, we'll use the tweet emotion dataset, which is a dataset of tweets labeled by one of four emotions:
# happiness, anger, sadness and optimism.

import numpy as np
from deepchecks.nlp.checks import PredictionDrift
from deepchecks.nlp.datasets.classification import tweet_emotion
np.random.seed(42)

train_ds, test_ds = tweet_emotion.load_data()

# Load precalculated model predictions:
train_preds, test_preds = tweet_emotion.load_precalculated_predictions(as_train_test=True)

#%%
# Let's see how our data looks like:

train_ds.head()

#%%
# Let's introduce drift into the data by dropping 50% of the "anger" tweets from the train dataset:

angry_tweets = np.argwhere(train_ds.label == 'anger').flatten()  # Get all angry tweets
# Select 50% of those to keep with the other tweets:
angry_tweets_to_ignore = np.random.choice(angry_tweets, size=len(angry_tweets) // 2, replace=False)
indices_to_keep = [x for x in range(len(train_ds)) if x not in angry_tweets_to_ignore]  # All indices to keep

# Recreate the dataset and predictions without the dropped samples:
train_ds = train_ds.copy(rows_to_use=indices_to_keep)
train_preds = train_preds[indices_to_keep]

#%%
# Run Check
# =========
#

check = PredictionDrift()
result = check.run(train_dataset=train_ds, test_dataset=test_ds,
                   train_predictions=train_preds, test_predictions=test_preds)

result

#%%
# We can see that we found drift in the distribution of the predictions, and that the drift is mainly in the "anger"
# class. This makes sense, as we dropped 50% of the "anger" tweets from the train dataset, and so the model is now
# predicting less "anger" tweets in the test dataset.
#
# The prediction drift check can also calculate drift on the probability of each class separately
# rather than the final predicted class.
# To force this behavior, set the ``drift_mode`` parameter to ``proba``.

# First let's get the probabilities for our data, instead of the predictions:
train_probas, test_probas = tweet_emotion.load_precalculated_predictions(pred_format='probabilities')
train_probas = train_probas[indices_to_keep]  # Filter the probabilities again

check = PredictionDrift(drift_mode='proba')
result = check.run(train_dataset=train_ds, test_dataset=test_ds,
                   train_probabilities=train_probas, test_probabilities=test_probas)

result

#%%
# This time, we can see there's small drift in each class. The "anger" class drift is actually probably caused by low
# sample size, and not by drift in the data itself, as we did not change the data within the class, but only changed
# the prevalence of the class in the data.

