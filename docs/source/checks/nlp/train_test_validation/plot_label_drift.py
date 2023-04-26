# -*- coding: utf-8 -*-
"""
.. _plot_nlp_label_drift:

Label Drift
**********************

This notebooks provides an overview for using and understanding the NLP label drift check.

**Structure:**

* `What Is Label Drift? <#what-is-label-drift>`__
* `Load Data <#load-data>`__
* `Run Check <#run-check>`__

What Is Label Drift?
========================
Drift is simply a change in the distribution of data over time, and it is
also one of the top reasons why machine learning model's performance degrades
over time.

Label drift is when drift occurs in the label itself.

For more information on drift, please visit our :doc:`drift guide </user-guide/general/drift_guide>`.

How Deepchecks Detects Label Drift
------------------------------------

This check detects label drift by using :ref:`univariate measures <drift_detection_by_univariate_measure>`
on the label.

"""

#%%
from deepchecks.nlp.datasets.classification import tweet_emotion
from deepchecks.nlp.checks import LabelDrift

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

# As there's natural drift in this dataset, we can expect to see some drift in the "optimism" label:

check = LabelDrift()
result = check.run(train_dataset=train_ds, test_dataset=test_ds)
result
