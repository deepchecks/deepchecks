# -*- coding: utf-8 -*-
"""
Quickstart - Getting Started
****************************

In order to run deepchecks for NLP all you need to have are the following for both your train and test data:

1. Your text data - a list of strings, each string is a single sample (can be a sentence, paragraph, document etc.).
2. Your labels - either a :ref:`Text Classification <nlp_supported_text_classification> label or a
   :ref:`Token Classification <nlp_supported_token_classification>` label.
3. Your models predictions (see :doc:`Supported Tasks </user-guide/nlp/supported_tasks>` for info on supported formats).

If you don't have deepchecks installed yet:

.. code:: python

    import sys
    !{sys.executable} -m pip install deepchecks[nlp] -U --quiet #--user

Some properties calculated by deepchecks.nlp require additional packages to be installed. You can
install them by running:

.. code:: python

    import sys
    !{sys.executable} -m pip install langdetect>=1.0.9 textblob>=0.17.1 -U --quiet #--user

Finally, we'll be using the CatBoost model in this guide, so we'll also need to install it:

.. code:: python
    import sys
    !{sys.executable} -m pip install catboost -U --quiet #--user

"""

#%%
# Load Data & Create TextData Objects
# ===================================
# For the purpose of this guide we'll use a small subset of the
# `tweet emotion <https://github.com/cardiffnlp/tweeteval>`__ dataset:

# Imports
from deepchecks.nlp import TextData
from deepchecks.nlp.datasets.classification import tweet_emotion

# Load Data
train, test = tweet_emotion.load_data(data_format='DataFrame')
train.head()

#%%
#
# We can see that we have the tweet text itself, the label (the emotion) and then some additional metadata columns.
#
# We can now create a :class:`TextData <deepchecks.nlp.TextData>` object for the train and test dataframes.
# This object is used to pass your data to the deepchecks checks.
#
# To create a TextData object, the only required argument is the text itself, but passing only the text
# will prevent multiple checks from running. In this example we'll pass the label as well and also provide
# metadata (the other columns in the dataframe) which we'll use later on in the guide. Finally, we'll also
# explicitly set the index.
#
# .. note::
#    The label column is optional, but if provided you must also pass the task_type argument, so that deepchecks
#    will know how to interpret the label column.
#

train = TextData(train.text, label=train['label'], task_type='text_classification',
                 index=train.index, metadata=train.drop(columns=['label', 'text']))
test = TextData(test.text, label=test['label'], task_type='text_classification',
                index=test.index, metadata=test.drop(columns=['label', 'text']))

#%%
# Building a Model
# ================
#
# In this example we'll train a very basic model for simplicity, using a CatBoostClassifier trained over the
# embeddings of the tweets. In this case these embeddings were created using the OpenAI GPT-3 model.
# If you want to reproduce this kind of basic model for your own task, you can calculate your own embeddings, or use
# our :func:`calculate_embeddings_for_text <deepchecks.nlp.utils.calculate_embeddings_for_text>`
# function to generate generic embeddings. Note that in order to run it you need either an OpenAI API key or have
# HuggingFace's transformers installed.

from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

# Load Embeddings and Split to Train and Test
embeddings = tweet_emotion.load_embeddings()
train_embeddings, test_embeddings = embeddings[train.index, :], embeddings[test.index, :]

model = CatBoostClassifier(max_depth=2, n_estimators=50, random_state=42)
model.fit(embeddings[train.index, :], train.label, verbose=0)
print(roc_auc_score(test.label,
                    model.predict_proba(embeddings[test.index, :]),
                    multi_class="ovr", average="macro"))

#%%
# Running Deepchecks
# ==================
#
# Now that we have our data and model, we can run our first checks!
#
# Performance Checks
# ------------------
#
# We'll start by running the
# :class:`TrainTestPredictionDrift <deepchecks.nlp.checks.model_evaluation.train_test_prediction_drift.TrainTestPredictionDrift>`
# check, which will let us know if there has been a significant change in the model's predictions between the train
# and test data. Such a change can imply that something has changed in the data distribution between the train and
# test data in a way that affects the model's predictions.
#
# We'll also add a condition to the check, which will make it fail if the drift score is higher than 0.1.

# Start by computing the predictions for the train and test data:
train_preds, train_probas = model.predict(embeddings[train.index, :]), model.predict_proba(embeddings[train.index, :])
test_preds, test_probas = model.predict(embeddings[test.index, :]), model.predict_proba(embeddings[test.index, :])

# Run the check
from deepchecks.nlp.checks import PredictionDrift

check = PredictionDrift().add_condition_drift_score_less_than(0.1)
result = check.run(train, test, train_predictions=list(train_preds), test_predictions=list(test_preds))

# Note: the result can be saved as html using suite_result.save_as_html()
# or exported to json using suite_result.to_json()
result.show()

#%%
# We can see that the check passed, and that the drift score is quite low.
#
# Next, we'll run the
# :class:`MetadataSegmentsPerformance <deepchecks.nlp.checks.model_evaluation.weak_segments_performance.MetadataSegmentsPerformance>`
# check, which will check the performance of the model on different segments of the metadata that we provided
# earlier when creating the ``TextData`` objects, and report back on any segments that have significantly lower
# performance than the rest of the data.
#

from deepchecks.nlp.checks import MetadataSegmentsPerformance

check = MetadataSegmentsPerformance()

result = check.run(test, predictions=list(test_preds), probabilities=test_probas)
result.show()

#%%
# As we can see, the check found a segment that has significantly lower performance than the rest of the data. In the
# first tab of the display we can see that there is a large segment of young Europeans that have significantly lower
# performance than the rest of the data. Perhaps there is some language gap here?
#
# Properties
# ^^^^^^^^^^
#
# Properties are one-dimension values that are extracted from the text. Among their uses, they can be used to
# segment the data, similar to the metadata segments that we saw in the previous check.
#
# Before we can run the
# :class:`PropertySegmentsPerformance <deepchecks.nlp.checks.model_evaluation.weak_segments_performance.PropertySegmentsPerformance>`
# check, we need to make sure that our TextData objects have the properties that we want to use. Properties can be added
# to the TextData objects in one of the following ways:
#
# 1. Calculated automatically by deepchecks. Deepchecks has a set of predefined properties that can be calculated
#    automatically. They can be added to the TextData object either by passing `properties='auto'` to the TextData
#    constructor, or by calling the
#    :meth:`calculate_default_properties <deepchecks.nlp.TextData.calculate_default_properties>` method anytime later.
# 2. You can calculate your own properties and then add them to the TextData object. This can be done by passing a
#    DataFrame of properties to the TextData `properties` argument, or by calling the
#    :meth:`set_properties <deepchecks.nlp.TextData.set_properties>` method anytime later with such a DataFrame. You
#

# Calculate properties
train.calculate_default_properties()
test.calculate_default_properties()

# Run the check
from deepchecks.nlp.checks import PropertySegmentsPerformance

check = PropertySegmentsPerformance()
result = check.run(test, predictions=list(test_preds), probabilities=test_probas)
result.show()

#%%
# As we can see, the check found a segment that has significantly lower performance than the rest of the data on our
# computed properties. For example, in the second tab of the display we can see that the model has significantly lower
# performance on tweets with positive sentiment.
#
# Data Integrity Checks
# ---------------------
#
# These previous checks were all about the model's performance. Now we'll run a check that attempts to find instances
# of shortcut learning - cases in which the label can be predicted by simple aspects of the data, which
# in many cases can be an indication that the model has used some information that won't generalize to the real world.
#
# This check is the
# :class:`PropertyLabelCorrelation <deepchecks.nlp.checks.data_integrity.property_label_correlation.PropertyLabelCorrelation>`
# check, which will check the correlation between the properties and the labels, and report back on any properties that
# have a high correlation with the labels.

from deepchecks.nlp.checks import PropertyLabelCorrelation

check = PropertyLabelCorrelation(n_top_features=10)
result = check.run(test)
result.show()

#%%
# In this case the check didn't find any properties that have a high correlation with the labels. Apart from the
# sentiment property, which is expected to have high relevance to the emotion of the tweet, the other properties
# have very low correlation to the label.
#
# You can find the full list of available NLP checks in the :mod:`api documentation <deepchecks.nlp.checks>`.

# sphinx_gallery_thumbnail_path = '_static/images/sphinx_thumbnails/nlp_quickstarts/getting_started.png'