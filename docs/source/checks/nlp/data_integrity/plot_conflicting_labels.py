# -*- coding: utf-8 -*-
"""
.. _nlp__conflicting_labels:

Conflicting Labels
******************

This notebook provides an overview for using and understanding the Conflicting Labels check:

**Structure:**

* `Why check for conflicting labels? <#why-check-for-conflicting-labels>`__
* `Create TextData <#create-textdata>`__
* `Run the Check <#run-the-check>`__
* `Define a Condition <#define-a-condition>`__

Why check for conflicting labels?
==================================

The ``ConflictingLabels`` check finds identical or nearly identical (see
`text normalization <#with-text-normalization>`__) samples in the dataset that have different labels. Conflicting labels
can lead to inconsistencies and confusion for the model during training. Identifying such samples can help in cleaning
the data and improving the model's performance.

Create TextData
===============

Lets create a simple dataset with some samples having conflicting labels.
"""

from deepchecks.nlp import TextData
from deepchecks.nlp.checks import ConflictingLabels

texts = [
    "Deep learning is a subset of machine learning.",
    "Deep learning is a subset of machine learning.",
    "Deep learning is a sub-set of Machine Learning.",
    "Deep learning is subset of machine learning",
    "Natural language processing is a subfield of AI.",
    "This is a unique text sample.",
    "This is another unique text.",
]

labels = [0, 1, 1, 0, 2, 2, 2]

dataset = TextData(texts, label=labels, task_type="text_classification")

# %%
# Run the Check
# =============

# Run the check without any text normalization
ConflictingLabels(
    ignore_case=False,
    remove_punctuation=False,
    normalize_unicode=False,
    remove_stopwords=False,
    ignore_whitespace=False,
).run(dataset)

# %%
# With Text Normalization
# -----------------------
# By default, ``ConflictingLabels`` check applies text normalization before identifying the conflicting labels.
# This includes case normalization, punctuation removal, Unicode normalization and stopwords removal.
# You can also customize the normalization as per your requirements:

ConflictingLabels(
    ignore_case=True, remove_punctuation=True, normalize_unicode=True, remove_stopwords=True, ignore_whitespace=True
).run(dataset)

# %%
# Of all the parameters in this example, ``ignore_whitespace`` is the only one set to ``False`` by default.
#
# Define a Condition
# ==================
#
# Now, we define a condition that enforces the ratio of samples with conflicting labels to be 0. A condition
# is deepchecks' way to validate model and data quality, and let you know if anything goes wrong.

check = ConflictingLabels()
check.add_condition_ratio_of_conflicting_labels_less_or_equal(0)
result = check.run(dataset)
result.show(show_additional_outputs=False)
