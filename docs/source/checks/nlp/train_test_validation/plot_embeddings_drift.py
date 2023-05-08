# -*- coding: utf-8 -*-
"""
.. _nlp__embeddings_drift:

Embeddings Drift
*******************

This notebooks provides an overview for using and understanding the embeddings
drift check.

**Structure:**

* `What Is Embeddings Drift? <#what-is-a-embeddings-drift>`__
* `Loading the Data <#loading-the-data>`__
* `Run the Check <#run-the-check>`__

What Is Embeddings Drift?
==============================

Drift is simply a change in the distribution of data over time, and it is
also one of the top reasons why machine learning model's performance degrades
over time.

In unstructured data such as text, we cannot measure the drift of the data directly, as there's no "distribution"
to measure. In order to measure the drift of the data, we can use the model's embeddings as a proxy for the data
distribution.

For more on embeddings, see the `NLP Embeddings Guide
<https://docs.deepchecks.com/stable/nlp/usage_guides/nlp_embeddings.html>`_.

How Deepchecks Detects Dataset Drift
------------------------------------

This check detects the embeddings drift by using :ref:`a domain classifier <drift_detection_by_domain_classifier>`.
Other methods to detect drift include :ref:`univariate measures <drift_detection_by_univariate_measure>`
which is used in other checks, such as :ref:`Feature Drift check <tabular__feature_drift>`.
"""

#%%
from deepchecks.nlp.datasets.classification import tweet_emotion
from deepchecks.nlp.checks import TextEmbeddingsDrift

#%%
# Load Data
# ==========
# For this example, we'll use the tweet emotion dataset, which is a dataset of tweets labeled by one of four emotions:
# happiness, anger, sadness and optimism.
train_ds, test_ds = tweet_emotion.load_data()

#%%
# Let's see how our data looks like:
train_ds.head()

#%%
# Run Check
# ===============================

#%%
# As there's natural drift in this dataset, we can expect to see some drift in the data:

check = TextEmbeddingsDrift()
result = check.run(train_dataset=train_ds, test_dataset=test_ds)
result

#%%
# Observing the results
# ----------------------
# We can see that the check found drift in the data. Moreover, we can investigate the drift by looking at the
# scatter plot, which is a 2D representation of the embeddings space. We can see that there are a few clusters
# in the graph where there are more tweets from the test dataset than the train dataset. This is a sign of drift
# in the data.
# By hovering over the points, we can see the actual tweets that are in the dataset, and see for example that
# there are clusters of tweets about motivational quotes, which are more common in the test dataset than the train
# dataset.
