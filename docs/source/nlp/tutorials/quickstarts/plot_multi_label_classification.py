# -*- coding: utf-8 -*-
"""
NLP Multi Label Classification Quickstart
*****************************************

In this quickstart guide, we will go over using the deepchecks NLP package to analyze and evaluate a text
multi label classification task. We will cover the following:

1. `Creating a TextData object and auto calculating properties <#setting-up>`__
2. `Running the built-in suites <#running-the-deepchecks-default-suites>`__
3. `Running individual checks <#running-individual-checks>`__

To run deepchecks for NLP, you need the following for both your train and test data:

1. Your text data - a list of strings, each string is a single sample (can be a sentence, paragraph, document, etc.).
2. Your labels and prediction in the :ref:`correct format <nlp__supported_text_classification>` (Optional).
3. :ref:`Metadata <nlp__metadata_guide>`, :ref:`Properties <nlp__properties_guide>`
   or :ref:`Embeddings <nlp__embeddings_guide>` for the provided text data (Optional).

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
`just dance <https://www.kaggle.com/datasets/renatojmsantos/just-dance-on-youtube>`__ comment analysis dataset.
A dataset containing comments, metadata and labels for a multilabel category classification use case on youtube comments.

"""

from deepchecks.nlp import TextData
from deepchecks.nlp.datasets.classification import just_dance_comment_analysis

data = just_dance_comment_analysis.load_data(data_format='DataFrame', as_train_test=False)
metadata_cols = ['likes', 'dateComment']
data.head(2)

# %%
# Create TextData Objects
# ------------------------
#
# Deepchecks' :ref:`TextData <nlp__textdata_object>` object contains the text samples, labels, and possibly
# also properties and metadata. It stores
# cache to save time between repeated computations and contains functionalities for input validations and sampling.

label_cols = data.drop(columns=['originalText'] + metadata_cols)
class_names = label_cols.columns.to_list()
dataset = TextData(data['originalText'], label=label_cols.to_numpy().astype(int),
                   task_type='text_classification', metadata=data[metadata_cols], categorical_metadata=[])

# %%
# Calculating Properties
# ----------------------
#
# Some of deepchecks' checks use properties of the text samples for various calculations. Deepcheck has a wide variety
# of such properties, some simple and some that rely on external models and are more heavy to run. In order for
# deepchecks' checks to be able to access the properties, they must be stored within the TextData object.

# properties can be either calculated directly by Deepchecks or
# imported from other sources in an :ref:`appropriate format <nlp__properties_guide>.

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# dataset.calculate_default_properties(include_long_calculation_properties=True, device=device)

properties = just_dance_comment_analysis.load_properties(as_train_test=False)
dataset.set_properties(properties, categorical_properties=['Language'])
dataset.properties.head(2)

# %%
# Running the deepchecks default suites
# =====================================
#
# Data Integrity
# --------------
# We will start by doing preliminary integrity check to validate the text formatting. It is recommended to do this step
# before your train and test/validation splits and model training as it may imply additional data
# engineering is required.
#
# We'll do that using the data_integrity pre-built suite. Note that we are limiting the number of samples to 1000
# in order to get quick high level overview of potential issues.

from deepchecks.nlp.suites import data_integrity

data_integrity_suite = data_integrity(n_samples=1000)
data_integrity_suite.run(dataset, model_classes=class_names)

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
# contain characters (specifically here emojis) that are
# unrecognized by the tokenizer. This is an important insight, as bert
# tokenizers are very common.
#
# Integrity #2: Conflicting Labels
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Looking at the Conflicting Labels check result (in the “Didn’t Pass” tab) we can
# see that there are 2 occurrences of duplicate samples that have different labels.
# This may suggest a more severe labeling error in the dataset which we would want to explore further.
#

# %%
# Train Test Validation
# ---------------------
# The next suite serves to validate our split and compare the two dataset. This suite is useful for when you already
# decided about your train and test/validation splits, but before training the model itself. It is recommended to run
# this suite during your model monitoring to verify the data hasn't change over time.
#
# For running the suite we need to split the data into train and test/validation sets. We'll use a predefined split
# our data based on comment dates.

from deepchecks.nlp.suites import train_test_validation

train_ds, test_ds = just_dance_comment_analysis.load_data(data_format='TextData', as_train_test=True,
                                                          include_embeddings=True, include_properties=True)
train_test_validation(n_samples=1000).run(train_ds, test_ds, model_classes=class_names)

# %%
# Train Test Validation #1: Properties Drift
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Based on the different properties we have calculated for the dataset, we can now search for
# properties whose distribution changes between the train and test datasets. Changes like this
# are especially important to look for when monitoring your model over time, as data drift
# is one of the top reasons why machine learning model’s performance degrades over time.
#
# In our case, we can see that the “% Special Characters”  and the "Formality" property have different distributions
# between train and test. Drilling further into the results, we can see that the language of the comments in the
# test set is much less formal and includes more special characters (possibly emojis?) than the train set.
# Since this change is quite significant, we may want to consider adding more informal comments containing
# special characters to the train set before training (or retraining) our model.
#
# Train Test Validation #2: Embedding Drift
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Similarly to the properties drift, we can also look for embedding drift between the train and test datasets.
# The benefit of using embedding on top of the properties is that they are able to detect semantic changes in the data.
#
# In our case, we see that are significant semantic differences between the train and test sets. Specifically,
# we can see some distinct segments that distinctly contain more
# samples from the train dataset or more sample from the test dataset. By hovering over the segments we can see
# text samples contained in each one and find the common ground.
#

# %%
# Model Evaluation
# ----------------
# The suite below is designed to be run after a model has been trained and requires model predictions which can be
# supplied via the relevant arguments in the ``run`` function.

train_preds, test_preds = just_dance_comment_analysis.\
    load_precalculated_predictions(pred_format='predictions', as_train_test=True)
train_probas, test_probas = just_dance_comment_analysis.\
    load_precalculated_predictions(pred_format='probabilities', as_train_test=True)

from deepchecks.nlp.suites import model_evaluation

suite = model_evaluation(n_samples=1000)
result = suite.run(train_ds, test_ds, train_predictions=train_preds, test_predictions=test_preds,
                   train_probabilities=train_probas, test_probabilities=test_probas, model_classes=class_names)
result.show()

# %%
# Model Eval #1: Train Test Performance
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# On the most superficial level, we can immediately see (in the "Did’t
# Pass" tab) that there has been significant degradation in the Recall on
# class “Pain and Discomfort”. Moreover, it seems there is a general deterioration in our model
# performance on the test set compared to the train set. This can be explained
# based on the data drift we saw in the previous suite.
#

# %%
# Running Individual Checks
# =========================
#
# Checks can also be run individually with relevant conditions. In this section,
# we'll show you how to do that while showcasing one of our
# most interesting checks - PropertySegmentPerformance.
#

from deepchecks.nlp.checks import PropertySegmentsPerformance

check = PropertySegmentsPerformance(segment_minimum_size_ratio=0.05)
check = check.add_condition_segments_relative_performance_greater_than(0.1)
result = check.run(test_ds, probabilities=test_probas)
result.show()

# %%
# In the display we can see some distinct property based segments that our model under performs on.
#
# By reviewing the results we can see that our model is performing poorly on samples that have a low level of
# Subjectivity, by looking at the "Subjectivity vs Average Sentence Length" tab
# We can see that the problem is even more severe on samples containing long sentences.
#
# In addition to the visual display, most checks also return detailed data describing the results. This data can be
# used for further analysis, create custom visualizations or to set custom conditions.
#

result.value['weak_segments_list'].head(3)

# %%
# You can find the full list of available NLP checks in the
# :mod:`nlp.checks api documentation ֿ <deepchecks.nlp.checks>`.
#
