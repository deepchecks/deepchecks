# -*- coding: utf-8 -*-
"""
.. _nlp__embeddings_drift:

Embeddings Drift
*******************

This notebooks provides an overview for using and understanding the embeddings
drift check.

**Structure:**

* `What Is Embeddings Drift? <#what-is-embeddings-drift>`__
* `Loading the Data <#load-data>`__
* `Run the Check <#run-check>`__

What Is Embeddings Drift?
==============================

Drift is simply a change in the distribution of data over time, and it is
also one of the top reasons why machine learning model's performance degrades
over time.

In unstructured data such as text, we cannot measure the drift of the data directly, as there's no "distribution"
to measure. In order to measure the drift of the data, we can use the model's embeddings as a proxy for the data
distribution.

For more on embeddings, see the :ref:`Text Embeddings Guide <nlp__embeddings_guide>`.

This detects embeddings drift by using :ref:`a domain classifier <drift_detection_by_domain_classifier>`.
For more information on drift, see the :ref:`Drift Guide <drift_user_guide>`.

How Does This Check Work?
=========================

This check detects the embeddings drift by using :ref:`a domain classifier <drift_detection_by_domain_classifier>`,
and uses the AUC score of the classifier as the basis for the measure of drift.
For efficiency, the check first reduces the dimensionality of the embeddings, and then trains the classifier on the
reduced embeddings. By default, the check uses UMAP for dimensionality reduction, but you can also use PCA by
setting the `dimension_reduction_method` parameter to `pca`.

The check also provides a scatter plot of the embeddings, which is a 2D representation of the embeddings space. This
is achieved by further reducing the dimensionality, using UMAP.

How To Use Embeddings in Deepchecks?
====================================
See how to calculate default embeddings or setting your own embeddings in the
:ref:`Embeddings Guide <using_nlp_embeddings_in_checks>`.
"""

# %%
from deepchecks.nlp.checks import TextEmbeddingsDrift
from deepchecks.nlp.datasets.classification import tweet_emotion

# %%
# Load Data
# ==========
#
# For this example, we'll use the tweet emotion dataset, which is a dataset of tweets labeled by one of four emotions:
# happiness, anger, sadness and optimism.
train_ds, test_ds = tweet_emotion.load_data()
train_embeddings, test_embeddings = tweet_emotion.load_embeddings(as_train_test=True)

# Set the embeddings in the datasets:
train_ds.set_embeddings(train_embeddings)
test_ds.set_embeddings(test_embeddings)

# %%
# Let's see how our data looks like:
train_ds.head()

# %%
# Run Check
# ===============================

# %%
# As there's natural drift in this dataset, we can expect to see some drift in the data:

check = TextEmbeddingsDrift()
result = check.run(train_dataset=train_ds, test_dataset=test_ds)
result

# %%
# Observing the results
# ----------------------
# We can see that the check found drift in the data. Moreover, we can investigate the drift by looking at the
# scatter plot, which is a 2D representation of the embeddings space. We can see that there are a few clusters
# in the graph where there are more tweets from the test dataset than the train dataset. This is a sign of drift
# in the data.
# By hovering over the points, we can see the actual tweets that are in the dataset, and see for example that
# there are clusters of tweets about motivational quotes, which are more common in the test dataset than the train
# dataset.
