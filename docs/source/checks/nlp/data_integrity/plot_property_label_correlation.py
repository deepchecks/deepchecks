# -*- coding: utf-8 -*-
"""
.. _plot_nlp_property_label_correlation:

Property Label Correlation
**************************

This notebook provides an overview for using and understanding the "Property Label Correlation" check.

**Structure:**

* `What Is The Purpose of the Check? <#what-is-the-purpose-of-the-check>`__
* `Run the Check <#run-the-check>`__

What Is The Purpose of the Check?
=================================
The check estimates for every :doc:`text property </user-guide/nlp/nlp_properties>`
(such as text length, language etc.) its ability to predict the label by itself.

This check can help find a potential bias in the dataset - the labels being strongly correlated with simple text
properties such as percentage of special characters, sentiment, toxicity and more.

This is a critical problem, sometimes referred to as shortcut learning, where the model is likely to learn this property
instead of the actual linguistic characteristics of each class, as it's easier to do so. In this case, the model will
show high performance on text taken in similar conditions, but will fail in the wild, where the simple properties
don't hold true.
This kind of correlation will likely stay hidden without this check until tested in the wild.

For example, in a classification dataset of true and false statements, if only true facts are written in detail,
and false facts are written in a short and vague manner, the model might learn to predict the label by the length
of the statement, and not by the actual content. In this case, the model will perform well on the training data,
and may even perform well on the test data, but will fail to generalize to new data.

The check is based on calculating the predictive power score (PPS) of each text
property. In simple terms, the PPS is a metric that measures how well can one feature predict another (in our case,
how well can one property predict the label).
For further information about PPS you can visit the `ppscore github <https://github.com/8080labs/ppscore>`__
or the following blog post: `RIP correlation. Introducing the Predictive Power Score
<https://towardsdatascience.com/rip-correlation-introducing-the-predictive-power-score-3d90808b9598>`__
"""

#%%
# Run the Check
# =============

from deepchecks.nlp.checks import PropertyLabelCorrelation
from deepchecks.nlp.datasets.classification import tweet_emotion

# For this example, we'll use the tweet emotion dataset, which is a dataset of tweets labeled by one of four emotions:
# happiness, anger, sadness and optimism.

# Load Data:
dataset = tweet_emotion.load_data(as_train_test=False)

#%%
# Let's see how our data looks like:
dataset.head()

#%%
# Now lets run the check:
result = PropertyLabelCorrelation().run(dataset)
result

#%%
# We can see that in our example of tweet emotion dataset, the label is correlated with the "sentiment" property,
# which makes sense, as the label is the emotion of the tweet, and the sentiment expresses whether the tweet is
# positive or negative.
