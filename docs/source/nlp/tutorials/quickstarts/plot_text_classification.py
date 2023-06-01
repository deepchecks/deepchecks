# -*- coding: utf-8 -*-
"""
.. _nlp__multiclass_quickstart:

Test NLP Classification Tasks - Quickstart
******************************************

Deepchecks NLP tests your models during model development/research and before deploying to production. Using our
testing package reduces model failures and saves tests development time. In this quickstart guide, you will learn how
to use the deepchecks NLP package to analyze and evaluate text
classification tasks. If you are interested in a multilabel classification task, you can
refer to our :ref:`Multilabel Quickstart <nlp__multilabel_quickstart>`. We will cover the following steps:

1. `Creating a TextData object and auto calculating properties <#setting-up>`__
2. `Running the built-in suites and inspecting the results <#running-the-deepchecks-default-suites>`__
3. `We'll spotlight two interesting checks - Embeddings drift and Under-Annotated Segments <#running-individual-checks>`__

To run deepchecks for NLP, you need the following for both your train and test data:

1. Your :ref:`text data <nlp__textdata_object>` - a list of strings, each string is a single sample
   (can be a sentence, paragraph, document, etc.).
2. Your labels - either a :ref:`Text Classification <nlp_supported_text_classification>` label or a
   :ref:`Token Classification <nlp_supported_token_classification>` label. These are not needed for checks that
   don't require labels (such as the Embeddings Drift check or most data integrity checks), but are needed for
   many other checks.
3. Your model's predictions (see :ref:`nlp__supported_tasks` for info on supported formats). These are needed only for
   the model related checks, shown in the `Model Evaluation <#model-evaluation>`__ section of this guide.

If you don't have deepchecks installed yet:

.. code:: python

    import sys
    !{sys.executable} -m pip install deepchecks[nlp] -U --quiet #--user

Some properties calculated by ``deepchecks.nlp`` require additional packages to be installed. You can
install them by running:

.. code:: python

    import sys
    !{sys.executable} -m pip install deepchecks[nlp-properties] -U --quiet #--user

Setting Up
==========

Load Data
---------
For the purpose of this guide, we'll use a small subset of the
`tweet emotion <https://github.com/cardiffnlp/tweeteval>`__ dataset. This dataset contains tweets and their
corresponding emotion - Anger, Happiness, Optimism, and Sadness.

"""

from deepchecks.nlp import TextData
from deepchecks.nlp.datasets.classification import tweet_emotion

train, test = tweet_emotion.load_data(data_format='DataFrame')
train.head()

# %%
#
# We can see that we have the tweet text itself, the label (the emotion) and then some additional metadata columns.
#
# Create a TextData Objects
# -------------------------
#
# We can now create a :ref:`TextData <nlp__textdata_object>` object for the train and test dataframes.
# This object is used to pass your data to the deepchecks checks.
#
# To create a TextData object, the only required argument is the text itself, but passing only the text
# will prevent multiple checks from running. In this example we'll pass the label and define the task type and finally
# define the :ref:`metadata columns <nlp__metadata_guide>` (the other columns in the dataframe) which we'll use later
# on in the guide.


train = TextData(train.text, label=train['label'], task_type='text_classification',
                 metadata=train.drop(columns=['label', 'text']))
test = TextData(test.text, label=test['label'], task_type='text_classification',
                metadata=test.drop(columns=['label', 'text']))

# %%
# Calculating Properties
# ----------------------
#
# Some of deepchecks' checks use properties of the text samples for various calculations. Deepcheck has a wide
# variety of such properties, some simple and some that rely on external models and are more heavy to run. In order
# for deepchecks' checks to be able to access the properties, they must be stored within the
# :ref:`TextData <nlp__textdata_object>` object. You can read more about properties in the
# :ref:`Property Guide <nlp__properties_guide>`.

# properties can be either calculated directly by Deepchecks
# or imported from other sources in appropriate format

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# train.calculate_builtin_properties(
#   include_long_calculation_properties=True, device=device
# )
# test.calculate_builtin_properties(
#   include_long_calculation_properties=True,  device=device
# )

# %%
# In this example though we'll use pre-calculated properties:

train_properties, test_properties = tweet_emotion.load_properties()

train.set_properties(train_properties, categorical_properties=['Language'])
test.set_properties(test_properties, categorical_properties=['Language'])

train.properties.head(2)

# %%
# Running the Deepchecks Default Suites
# =====================================
#
# Deepchecks comes with a set of pre-built suites that can be used to run a set of checks on your data, alongside
# with their default conditions and thresholds. You can read more about customizing and creating your own suites in the
# :ref:`Customizations Guide <general__customizations>`. In this guide we'll be using 3 suites - the data integrity
# suite, the train test validation suite and the model evaluation suite. You can also run all the checks at once using
# the :mod:`full_suite <deepchecks.nlp.suites>`.
#
# Data Integrity
# --------------
# We will start by doing preliminary integrity check to validate the text formatting. It is recommended to do this step
# before model training as it may imply additional data engineering is required.
#
# We'll do that using the :mod:`data_integrity <deepchecks.nlp.suites>` pre-built suite.

from deepchecks.nlp.suites import data_integrity

data_integrity_suite = data_integrity()
data_integrity_suite.run(train, test)

# %%
# Integrity #1: Unknown Tokens
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# First up (in the “Didn't Pass” tab) we see that the Unknown Tokens check
# has returned a problem.
#
# Looking at the result, we can see that it assumed (by default) that
# we’re going to use the bert-base-uncased tokenizer for our NLP model,
# and that if that’s the case there are many words in the dataset that
# contain characters (such as emojis, or Korean characters) that are
# unrecognized by the tokenizer. This is an important insight, as bert
# tokenizers are very common. You can configure the tokenizer used by
# this check by passing the tokenizer to the check’s constructor, and can
# also configure the threshold for the percent of unknown tokens allowed by
# modifying the checks condition.
#
# Integrity #2: Text Outliers
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# In the “Other” tab, Looking at the Text Outlier check result we can
# derive several insights by hovering over the different values and inspecting the outlier texts:
#
# 1. hashtags (‘#…’) are usually several words
#    written together without spaces - we might consider splitting them
#    before feeding the tweet to a model
# 2. In some instances users
#    deliberately misspell words, for example ‘!’ instead of the letter ‘l’
#    or ‘okayyyyyyyyyy’.
# 3. The majority of the data is in English but not
#    all. If we want a classifier that is multilingual we should collect
#    more data, otherwise we may consider dropping tweets in other languages
#    from our dataset before training our model.
#
# Integrity #3: Property-Label Correlation (Shortcut Learning)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# In the "Passed" tab we can see tha Property-Label Correlation check, that verifies the data does not contain
# any shortcuts the model can fixate on during the learning process. In
# our case we can see no indication that this problem exists in our
# dataset. For more information about shortcut learning see:
# https://towardsdatascience.com/shortcut-learning-how-and-why-models-cheat-1b37575a159

# %%
# Train Test Validation
# ---------------------
#
# The next suite, the :mod:`train_test_validation <deepchecks.nlp.suites>` suite serves to validate our split and
# compare the two dataset. These splits can be either you training and val / test sets, in which case you'd want to run
# this suite after the split was made but before training, or for example your training and inference data, in which
# case the suite is useful for validating that the inference data is similar enough to the training data.

from deepchecks.nlp.suites import train_test_validation

train_test_validation().run(train, test)

# %%
# Label Drift
# ~~~~~~~~~~~
#
# This check, appearing in the "Didn't Pass" tab, lets us see that we have some significant change in the
# distribution of the label - the label “optimism” is suddenly way more common in the test dataset, while other
# labels declined. This happened because we split on time, so the topics covered by the tweets in the test dataset
# may correspond to specific trends or events that happened later in time. Let’s investigate!

# %%
# Model Evaluation
# ----------------
#
# The suite below, the :mod:`model_evaluation <deepchecks.nlp.suites>` suite, is designed to be run after a model has
# been trained and requires model predictions which can be supplied via the relevant arguments in the ``run`` function.

train_preds, test_preds = tweet_emotion.load_precalculated_predictions(
    pred_format='predictions', as_train_test=True)
train_probas, test_probas = tweet_emotion.load_precalculated_predictions(
    pred_format='probabilities', as_train_test=True)

from deepchecks.nlp.suites import model_evaluation

result = model_evaluation().run(train, test,
                                train_predictions=train_preds,
                                test_predictions=test_preds,
                                train_probabilities=train_probas,
                                test_probabilities=test_probas)
result.show()

# %%
# OK! We have many important issues being surfaced by this suite. Let’s dive into the individual checks:
#
# Model Eval #1: Train Test Performance
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We can immediately see in the "Didn't Pass" tab that there has been significant degradation in the Recall on
# class “optimism”. This is very likely a result of the severe label drift we saw after running the previous suite.
#
# Model Eval #2: Segment Performance
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Also in the "Didn't Pass" tab we can see the two segment performance checks - Property Segment Performance and
# Metadata Segment Performance. These use the :ref:`metadata columns <nlp__metadata_guide>` of user related
# information OR our :ref:`calculated properties <nlp__properties_guide>` to try and **automatically**
# detect significant data segments on which our model performs badly.
#
# In this case we can see that both checks have found issues in the test
# dataset:
#
# 1. The Property Segment Performance check has found that we’re
#    getting very poor results on low toxicity samples. That probably means
#    that our model is using the toxicity of the text to infer the “anger”
#    label, and is having a harder problem with other, more benign text
#    samples.
# 2. The Metadata Segment Performance check has found that we
#    have predicting correct results on new users from the Americas. That’s
#    5% of our dataset so we better investigate that further.
#
# You'll note that these two issues occur only in the test data, and so the results of these checks for the
# training data appear in the "Passed" tab.
#
# Model Eval #3: Prediction Drift
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We note that the Prediction Drift (here in the “Passed” tab) shows no
# issue. Given that we already know that there is significant Label Drift,
# this means we have Concept Drift - the labels corresponding to our
# samples have changed, while the model continues to predict the same
# labels. You can learn more about the different types of drift and how deepchecks detects them in our
# :ref:`Drift Guide <drift_user_guide>`.

# %%
# Running Individual Checks
# =========================
#
# Checks can also be run individually. In this section, we'll show two of the more interesting checks and how you can
# run them stand-alone and add conditions to them. You can learn more about customizing suites, checks and conditions
# in our :ref:`Customizations Guide <general__customizations>`.
#
# Embeddings Drift
# ----------------
#
# In order to run the :ref:`Embeddings Drift <nlp__embeddings_drift>` check
# you must have text embeddings loaded to both datasets. You can read more about using embeddings in deepchecks NLP in
# our :ref:`Embeddings Guide <nlp__embeddings_guide>`.
#
# In this example, we have the embeddings already
# pre-calculated:

from deepchecks.nlp.datasets.classification.tweet_emotion import load_embeddings

train_embeddings, test_embeddings = load_embeddings()

train.set_embeddings(train_embeddings)
test.set_embeddings(test_embeddings)

# %%
# You can also calculate the embeddings using deepchecks, either using an
# open-source sentence-transformer or using Open AI’s embedding API.

# train.calculate_builtin_embeddings()
# test.calculate_builtin_embeddings()

# %%
#

from deepchecks.nlp.checks import TextEmbeddingsDrift

check = TextEmbeddingsDrift()
res = check.run(train, test)
res.show()

# %%
# Here we can see some clusters that distinctly contain more
# samples from train or more sample for test. For example, if we look at
# the greenish cluster in the middle (by hovering on the samples and reading the tweets) we see it’s full of
# inspirational quotes and sayings, and belongs mostly to the test dataset. That is the
# source of the drastic increase in optimistic labels!
#
# There are of course also other note-worthy clusters, such as the greenish cluster on the right that contains tweets
# about a terror attack in Bangladesh, which belongs solely to the test data.

# %%
# Under Annotated Segments
# ------------------------
#
# Another note-worthy segment is the
# :ref:`Under Annotated Segments <nlp__under_annotated_property_segments>` check,
# which explores our data and automatically identifies segments where the data
# is under-annotated - meaning that the ratio of missing labels is higher.
# To this check we’ll also add a condition that will fail in case that
# an under-annotated segment of significant size is found.

from deepchecks.nlp.checks import UnderAnnotatedPropertySegments
test_under = tweet_emotion.load_under_annotated_data()

check = UnderAnnotatedPropertySegments(
    segment_minimum_size_ratio=0.1
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