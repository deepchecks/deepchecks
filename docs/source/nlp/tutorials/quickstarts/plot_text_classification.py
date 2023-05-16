# -*- coding: utf-8 -*-
"""
Test NLP Classification Tasks - Quickstart
******************************************

In this quickstart guide, we will go over using the deepchecks NLP package to analyze and evaluate text
classification tasks. We will cover the following:

1. Creating a TextData object and auto calculating properties
2. Running the built-in suites
3. Check spotlight - Embeddings drift and Under-Annotated Segments

To run deepchecks for NLP, you need the following for both your train and test data:

1. Your text data - a list of strings, each string is a single sample (can be a sentence, paragraph, document, etc.).
2. Your labels - either a :ref:`Text Classification <nlp_supported_text_classification>` label or a
   :ref:`Token Classification <nlp_supported_token_classification>` label.
3. Your models predictions (see :ref:`nlp__supported_tasks` for info on supported formats).

If you don't have deepchecks installed yet:

.. code:: python

    import sys
    !{sys.executable} -m pip install deepchecks[nlp] -U --quiet #--user

Some properties calculated by ``deepchecks.nlp`` require additional packages to be installed. You can
install them by running:

.. code:: python

    import sys
    !{sys.executable} -m pip install [nlp-properties] -U --quiet #--user

Setting Up
==========

Load Data
---------
For the purpose of this guide, we'll use a small subset of the
`tweet emotion <https://github.com/cardiffnlp/tweeteval>`__ dataset:

"""

from deepchecks.nlp import TextData
from deepchecks.nlp.datasets.classification import tweet_emotion

train, test = tweet_emotion.load_data(data_format='DataFrame')
train.head()

# %%
# Create TextData Objects
# ------------------------
#
# Deepchecks' TextData object contains the text samples, labels, and possibly also properties and metadata. It stores
# cache to save time between repeated computations and contains functionalities for input validations and sampling.


train = TextData(train.text, label=train['label'], task_type='text_classification',
                 metadata=train.drop(columns=['label', 'text']))
test = TextData(test.text, label=test['label'], task_type='text_classification',
                metadata=test.drop(columns=['label', 'text']))

# %%
# Calculating Properties
# ----------------------
#
# Some of deepchecks' checks use properties of the text samples for various calculations. Deepcheck has a wide variety
# of such properties, some simple and some that rely on external models and are more heavy to run. In order for
# deepchecks' checks to be able to access the properties, they must be stored within the TextData object.

# properties can be either calculated directly by Deepchecks or imported for other sources in appropriate format

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# train.calculate_default_properties(include_long_calculation_properties=True, device=device)
# test.calculate_default_properties(include_long_calculation_properties=True,  device=device)

train_properties, test_properties = tweet_emotion.load_properties()

train.set_properties(train_properties, categorical_properties=['Language'])
test.set_properties(test_properties, categorical_properties=['Language'])

train.properties.head(2)

# %%
# Running the deepchecks default suites
# =====================================
#
# Data Integrity
# --------------
# We will start by doing preliminary integrity check to validate the text formatting. It is recommended to do this step
# before model training as it may imply additional data engineering is required.
#
# We'll do that using the data_integrity pre-built suite.

from deepchecks.nlp.suites import data_integrity

data_integrity_suite = data_integrity()
data_integrity_suite.run(train, test)

# %%
# Integrity #1: Unknown Tokens
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# First up (in the “Didn’t Pass” tab) we see that the Unknown Tokens check
# has returned a problem.
#
# Looking at the result, we can see that it assumed (by default) that
# we’re going to use the bert-base-uncased tokenizer for our NLP model,
# and that if that’s the case there are many words in the dataset that
# contain characters (such as emojies, or Korean characters) that are
# unrecognized by the tokenizer. This is an important insight, as bert
# tokenizers are very common.
#
# Integrity #2: Text outliers
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Looking at the Text Outlier check result (in the “Other” tab) we can
# derive several insights: 1. hashtags (‘#…’) are usually several words
# written together without spaces - we might consider splitting them
# before feeding the tweet to a model 2. In some instances users
# deliberately misspell words, for example ‘!’ instead of the letter ‘l’
# or ‘okayyyyyyyyyy’ 3. The majority of the data is in English but not
# all. If we want a classifier that is multi lingual we should collect
# more data, otherwise we may consider dropping tweets in other languages
# from our dataset before training our model.
#
# Integrity #3: Property-Label Correlation (Shortcut Learning)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The Property-Label Correlation check verifies the data does not contain
# any shortcuts the model can fixate on during the learning process. In
# our case we can see no indication that this problem exists in our
# dataset For more information about shortcut learning see:
# https://towardsdatascience.com/shortcut-learning-how-and-why-models-cheat-1b37575a159

# %%
# Train Test Validation
# ---------------------
# The next suite serves to validate our split and compare the two dataset. This suite is useful for when you already
# decided about your train and test/validation splits, but before training the model itself.

from deepchecks.nlp.suites import train_test_validation

train_test_validation().run(train, test)

# %%
# Label Drift
# ~~~~~~~~~~~
#
# We can see that we have some significant change in the distribution of
# the label - the label “optimism” is suddenly way more common in the test
# dataset, while other labels declined. This happened because we split on
# time, so the topics covered by the tweets in the test dataset may
# correspond to specific trends or events that happened later in time.
# Let’s investigate!

# %%
# Model Evaluation
# ----------------
# The suite below is designed to be run after a model has been trained and requires model predictions which can be
# supplied via the relevant arguments in the ``run`` function.

train_preds, test_preds = tweet_emotion.load_precalculated_predictions(pred_format='predictions', as_train_test=True)
train_probas, test_probas = tweet_emotion.load_precalculated_predictions(pred_format='probabilities',
                                                                         as_train_test=True)

from deepchecks.nlp.suites import model_evaluation

result = model_evaluation().run(train, test, train_predictions=train_preds, test_predictions=test_preds,
                                train_probabilities=train_probas, test_probabilities=test_probas)
result.show()

# %%
# OK! We have many important issues being surfaced by this suite. Let’s
# dive into the individual checks:
#
# Model Eval #1: Train Test Performance
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# On the most superficial level, we can immediately see (in the "Did’t
# Pass" tab) that there has been significant degradation in the Recall on
# class “optimism”. This follows from the severe label drift we saw after
# running the previous suite.
#
# Model Eval #2: Segment Performance
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The two segment performance checks - Property Segment Performance and
# Metadata Segment Performance, use the metadata columns of user related
# information OR our calculated properties to try and **automatically**
# detect significant data segments on which our model performs badly.
#
# In this case we can see that both checks have found issues in the test
# dataset: 1. The Property Segment Performance check has found that we’re
# getting very poor results on low toxicity samples. That probably means
# that our model is using the toxicity of the text to infer the “anger”
# label, and is having a harder problem with other, more benign text
# samples. 2. The Metadata Segment Performance check has found that we
# have predicting correct results on new users from the Americas. That’s
# 5% of our dataset so we better investigate that further.
#
# Model Eval #3: Prediction Drift
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We note that the Prediction Drift (here in the “Passed” tab) shows no
# issue. Given that we already know that there is significant Label Drift,
# this means we have Concept Drift - the labels corresponding to our
# samples have changed, while the model continues to predict the same
# labels.

# %%
# Running Individual Checks
# =========================
#
# Checks can also be run individually. In this section, we'll show two of the more interesting checks and how you can
# run them stand-alone.
#
# Embeddings Drift
# ----------------

from deepchecks.nlp.datasets.classification.tweet_emotion import load_embeddings

train_embeddings, test_embeddings = load_embeddings()

train.set_embeddings(train_embeddings)
test.set_embeddings(test_embeddings)

from deepchecks.nlp.checks import TextEmbeddingsDrift

check = TextEmbeddingsDrift()
res = check.run(train, test)
res.show()

# %%
# Here we can see some distinct segments that distinctly contain more
# samples from train or more sample for test. For example, if we look at
# the cluster in the top-left corner we see it’s full of inspirational
# quotes and saying, and belongs mostly to the test dataset. That is the
# source of the drastic increase in optimistic labels!
#
# There are some other note-worthy segments, such as the “tail” segment in
# the middle left that contains tweets about a terror attack in Bangladesh
# (and belongs solely to the test data), or a cluster on the bottom right
# that discusses a sports event that probably happened strictly in the
# training dataset.

# %%
# Under Annotated Segments
# ------------------------
#
# Another note-worth segment is the Under Annotated Segment check, which
# explores our data and automatically identifies segments where the data
# is under-annotated - meaning that the ratio of missing labels is higher.
# To this check we’ll also add a condition that will alert us in case that
# a significant under-annotated segment is found.

from deepchecks.nlp.checks import UnderAnnotatedPropertySegments
test_under = tweet_emotion.load_under_annotated_data()
check = UnderAnnotatedPropertySegments(segment_minimum_size_ratio=0.1
                                       ).add_condition_segments_relative_performance_greater_than()
check.run(test_under)

# %%
# For example, here the check detected that we have a lot of lacking annotations for samples that are informal and
# not very fluent. May it be the case that our annotators have a problem annotating these samples and prefer not to
# deal with them? If these samples are important for use, we may have to put special focus on annotating this segment.
#
# .. note::
#
#     You can find the full list of available NLP checks in the :mod:`nlp.checks api documentation ֿ
#     <deepchecks.nlp.checks>`.

# sphinx_gallery_thumbnail_path = '_static/images/sphinx_thumbnails/nlp_quickstarts/getting_started.png'