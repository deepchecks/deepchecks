# -*- coding: utf-8 -*-
"""
.. _nlp__token_classification_quickstart:

Test NLP Token Classification Tasks - Quickstart
************************************************

In this quickstart guide, we will give a brief example of using the deepchecks NLP package to analyze and evaluate
token classification tasks. A token classification task is a case in which we wish to give a specific label for each
token (usually a word or a part of a word), rather than assigning a class or classes for the text as a whole. For a
more complete example showcasing the range of checks and capabilities of the NLP package, refer to our
:ref:`Multiclass Quickstart <nlp__multiclass_quickstart>`. We will cover the following steps:

1. `Creating a TextData object and auto calculating properties <#setting-up>`__
2. `Running checks <#running-checks>`__

To run deepchecks for token classification, you need the following for both your train and test data:

1. Your tokenized text dataset - a list containing lists of strings, each string is a single token within the sample,
   where a sample can be a sentence, paragraph, document and so on.
2. Your labels - a :ref:`Token Classification <nlp__supported_token_classification>` label. These are not needed for
   checks that don't require labels (such as the Embeddings Drift check or most data integrity checks), but are needed
   for many other checks.
3. Your model's predictions (see :ref:`nlp__supported_tasks` for info on supported formats). These are needed only for
   the model related checks, shown in the `Model Evaluation <#running-checks>`__ check in this guide.

If you don't have deepchecks installed yet:

.. code:: python

    import sys
    !{sys.executable} -m pip install deepchecks[nlp] -U --quiet #--user

Some properties calculated by ``deepchecks.nlp`` require additional packages to be installed. You can
also install them by running:

.. code:: python

    import sys
    !{sys.executable} -m pip install deepchecks[nlp-properties] -U --quiet #--user

Setting Up
==========

Load Data
---------
For the purpose of this guide, we'll use a small subset of the
`SCIERC <http://nlp.cs.washington.edu/sciIE/>`__ dataset:

"""
from pprint import pprint
from deepchecks.nlp import TextData
from deepchecks.nlp.datasets.token_classification import scierc_ner

train, test = scierc_ner.load_data(data_format='Dict')
pprint(train['text'][0][:10])
pprint(train['label'][0][:10])

# %%
#
# The SCIERC dataset is a dataset of scientific articles with annotations for named entities, relations and
# coreferences.
# In this example we'll only use the named entity annotations, which are the labels we'll use for our token
# classification task.
# We can see that we have the article text itself, and the labels for each token in the text in the
# :ref:`IOB format <nlp__supported_token_classification>`.
#
# Create a TextData Object
# -------------------------
#
# We can now create a :ref:`TextData <nlp__textdata_object>` object for the train and test dataframes.
# This object is used to pass your data to the deepchecks checks.
#
# To create a TextData object, the only required argument is the tokenized text itself. In most cases we'll want to
# pass labels as well, as they are needed in order to calculate many checks. In this example we'll pass the label and
# define the task type.


train = TextData(tokenized_text=train['text'], label=train['label'], task_type='token_classification')
test = TextData(tokenized_text=test['text'], label=test['label'], task_type='token_classification')

# %%
# Calculating Properties
# ----------------------
#
# Some of deepchecks' checks use properties of the text samples for various calculations. Deepcheck has a wide
# variety of such properties, some simple and some that rely on external models and are more heavy to run. In order
# for deepchecks' checks to be able to use the properties, they must be added to the
# :ref:`TextData <nlp__textdata_object>` object, usually by calculating them. You can read more about properties in the
# :ref:`Property Guide <nlp__properties_guide>`.

# properties can be either calculated directly by Deepchecks
# or imported from other sources in appropriate format

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# train.calculate_builtin_properties(
#   include_long_calculation_properties=True, device=device
# )
# test.calculate_builtin_properties(
#   include_long_calculation_properties=True, device=device
# )

# %%
# In this example though we'll use pre-calculated properties:

train_properties, test_properties = scierc_ner.load_properties()

train.set_properties(train_properties, categorical_properties=['Language'])
test.set_properties(test_properties, categorical_properties=['Language'])

train.properties.head(2)

# %%
# Running Checks
# ==============
#
# Train Test Performance
# ----------------------
#
# Once the :ref:`TextData <nlp__textdata_object>` object is ready, we can run the checks. We'll start by running
# the :ref:`TrainTestPerformance <nlp__train_test_performance>` check, which compares the performance of the model on
# the train and test sets. For this check, we'll need to pass the model's predictions on the train and test sets, also
# provided in the format of an IOB annotation per token in the tokenized text.
#
# We'll also define a condition for the check with the default threshold value. You can learn more about customizing
# checks and conditions, as well as defining suites of checks in our
# :ref:`Customizations Guide <general__customizations>`

train_preds, test_preds = scierc_ner.load_precalculated_predictions()

from deepchecks.nlp.checks import TrainTestPerformance
check = TrainTestPerformance().add_condition_train_test_relative_degradation_less_than()
result = check.run(train, test, train_predictions=train_preds, test_predictions=test_preds)
result

# %%
# We can see that the model performs better on the train set than on the test set, which is expected. We can also note
# specifically that the recall for class "OtherScientificTerm" has declined significantly on the test set, which is
# something we might want to investigate further.
#
# Embeddings Drift
# ----------------
#
# The :ref:`EmbeddingsDrift <nlp__embeddings_drift>` check compares the embeddings of the train and test sets. In
# order to run this check you must have text embeddings loaded to
# both datasets. You can read more about using embeddings in deepchecks NLP in our
# :ref:`Embeddings Guide <nlp__embeddings_guide>`. In this example, we have the embeddings already pre-calculated:

train_embeddings, test_embeddings = scierc_ner.load_embeddings()


train.set_embeddings(train_embeddings)
test.set_embeddings(test_embeddings)

# %%
# You can also calculate the embeddings using deepchecks, either using an
# open-source sentence-transformer or using Open AI’s embedding API.

# train.calculate_builtin_embeddings()
# test.calculate_builtin_embeddings()

# %%

from deepchecks.nlp.checks import TextEmbeddingsDrift

check = TextEmbeddingsDrift()
res = check.run(train, test)
res.show()

# %%
# The check shows the samples from the train and test datasets as points in the 2-dimensional reduced embedding
# space. We can see some distinct segments  - in the upper left corner we can notice (by hovering on the samples and
# reading the abstracts) that these are papers about computer vision, while the bottom right corner is mostly about
# Natural Language Processing. We can also see that although there isn't significant drift between the train and test,
# the training dataset has a bit more samples from the NLP domain, while the test set has more samples from the
# computer vision domain.
#
# .. note::
#
#     You can find the full list of available NLP checks in the :mod:`nlp.checks api documentation ֿ
#     <deepchecks.nlp.checks>`.
