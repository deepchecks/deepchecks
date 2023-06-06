# -*- coding: utf-8 -*-
"""
.. _nlp__text_property_outliers:

Text Property Outliers
=======================

This notebooks provides an overview for using and understanding the text property
outliers check, used to detect outliers in simple text properties in a dataset.

**Structure:**

* `Why Check for Outliers? <#why-check-for-outliers>`__
* `How Does the Check Work? <#how-does-the-check-work>`__
* `Which Text Properties Are Used? <#which-text-properties-are-used>`__
* `Run the Check <#run-the-check>`__


Why Check for Outliers?
-----------------------
Examining outliers may help you gain insights that you couldn't have reached from taking an aggregate look or by
inspecting random samples. For example, it may help you understand you have some corrupt samples (e.g.
texts without spaces between words), or samples you didn't expect to have (e.g. texts in Norwegian instead of English).
In some cases, these outliers may help debug some performance discrepancies (the model can be excused for failing on
a totally blank text). In more extreme cases, the outlier samples may indicate the presence of samples interfering with
the model's training by teaching the model to fit "irrelevant" samples.


How Does the Check Work?
------------------------
Ideally we would like to directly find text samples which are outliers, but this is computationally expensive and does not
have a clear and explainable results. Therefore, we use text properties in order to find outliers (such as text length,
average word length, language etc.) which are much more efficient to compute, and each outlier is easily explained.

* For numeric properties (such as "percent of special characters"), we use
  `Interquartile Range <https://en.wikipedia.org/wiki/Interquartile_range#Outliers>`_ to define our upper and lower
  limit for the properties' values.
* For categorical properties (such as "language"), we look for a "sharp drop" in the category distribution to
  define our lower limit for the properties' values. This method is based on the assumption that the distribution of
  categories in the dataset is "smooth" and differences in the commonality of categories are gradual.
  For example, in a clean dataset, if the distribution of English texts is 80%, the distribution of the next most
  common language would be of similar scale (e.g. 10%) and so forth. If we find a category that has a much lower
  distribution than the rest, we assume that this category and even smaller categories are outliers.

Which Text Properties Are Used?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
By default the checks uses the properties that where calculated for the train and test datasets, which by default are
the built-in text properties. It's also possible to replace the default properties with custom ones. For the list
of the built-in text properties and explanation about custom properties refer to :ref:`NLP properties
<nlp__properties_guide>`.

.. note::

    If a property was not calculated for a sample (for example, if it applies only to English samples and the sample
    is in another language), it will contain a nan value and will be ignored when calculating the outliers.

"""

#%%
# Run the Check
# -------------
# For this example, we'll use the tweet emotion dataset, which is a dataset of tweets labeled by one of four emotions:
# happiness, anger, sadness and optimism.

from deepchecks.nlp.checks import TextPropertyOutliers
from deepchecks.nlp.datasets.classification import tweet_emotion

dataset = tweet_emotion.load_data(as_train_test=False)

check = TextPropertyOutliers()
result = check.run(dataset)
result

#%%
# Observe Graphic Result
# ^^^^^^^^^^^^^^^^^^^^^^
# In this example, we can find many tweets that are outliers - For example, in the "average word length" property,
# we can see that there are tweets with a very large average word length, which is is usually because of missing spaces
# in the tweet itself, or the fact that tweeter hashtags remained in the data and they don't contain spaces. This
# could be problematic for the model, as it cannot coprehend the hashtags as words, and it may cause the model to
# fail on these tweets.
