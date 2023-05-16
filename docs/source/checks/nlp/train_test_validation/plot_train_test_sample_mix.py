# -*- coding: utf-8 -*-
"""
.. _nlp__train_test_samples_mix:

Train-Test Samples Mix
************************

This notebook provides an overview for using and understanding the train-test samples mix check:

**Structure:**

* `Why check for train-test samples mix? <#why-check-for-train-test-samples-mix>`__
* `Create TextData for Train and Test Sets <#create-textdata-for-train-and-test-sets>`__
* `Run the Check <#run-the-check>`__
* `Define a Condition <#define-a-condition>`__

Why check for train-test samples mix?
======================================
The ``TrainTestSamplesMix`` check finds instances of identical or very similar samples in both the
train and test datasets. If such samples are present unintentionally, it may lead to data leakage, which
can result in overly optimistic model performance estimates during evaluation. Identifying and addressing
such issues is crucial to ensure the model performs well on unseen data.

Create TextData for Train and Test Sets
========================================

Let's create train and test datasets with some overlapping and similar text samples.
"""

from deepchecks.nlp.checks import TrainTestSamplesMix
from deepchecks.nlp import TextData

train_texts = [
    "Deep learning is a subset of machine learning.",
    "Deep learning is a subset of machine learning.",
    "Deep learning is a sub-set of Machine Learning.",
    "Natural language processing is a subfield of AI.",]

test_texts = [
    "Deep learning is a subset of machine learning.",
    "Deep learning is subset of machine learning",
    "Machine learning is a subfield of AI.",
    "This is a unique text sample in the test set.",
    "This is another unique text in the test set.",
]

train_dataset = TextData(train_texts)
test_dataset = TextData(test_texts)

#%%
# Run the Check
# =============

# Run the check without any text normalization
check = TrainTestSamplesMix(
    ignore_case=False,
    remove_punctuation=False,
    normalize_unicode=False,
    remove_stopwords=False,
    ignore_whitespace=False
)
result = check.run(train_dataset, test_dataset)
result.show()

# %%
# With Text Normalization
# -----------------------
#
# By default, ``TrainTestSamplesMix`` check applies text normalization before identifying the duplicates.
# This includes case normalization, punctuation removal, Unicode normalization and stopwords removal.
# You can also customize the normalization as per your requirements:

check = TrainTestSamplesMix(
    ignore_case=True,
    remove_punctuation=True,
    normalize_unicode=True,
    remove_stopwords=True,
    ignore_whitespace=True
)
result = check.run(train_dataset, test_dataset)
result.show()

# %%
# Of all the parameters in this example, ``ignore_whitespace`` is the only one set to ``False`` by default.
#
# Define a Condition
# ==================
#
# Now, we define a condition that enforces the ratio of duplicates to be 0. A condition
# is deepchecks' way to validate model and data quality, and let you know if anything
# goes wrong.

check = TrainTestSamplesMix()
check.add_condition_duplicates_ratio_less_or_equal(0)
result = check.run(train_dataset, test_dataset)
result.show(show_additional_outputs=False)
