# -*- coding: utf-8 -*-
"""
.. _text__data_duplicates:

Text Data Duplicates
********************

This notebook provides an overview for using and understanding the text data duplicates check:

**Structure:**

* `Why check for text data duplicates? <#why-check-for-text-data-duplicates>`__
* `Create TextData <#create-textdata>`__
* `Run the Check <#run-the-check>`__
* `Define a Condition <#define-a-condition>`__

Why check for text data duplicates?
===================================
The ``TextDuplicates`` check finds multiple instances of identical or very similar samples in the
Dataset. Duplicate samples increase the weight the model gives to those samples.
If these duplicates are there intentionally (e.g. as a result of intentional
oversampling, or due to the dataset's nature it has identical-looking samples)
this may be valid, however if this is a hidden issue we're not expecting to occur,
it may be an indicator for a problem in the data pipeline that requires attention.

Create TextData
===============

Let's create a simple dataset with some duplicate and similar text samples.
"""

from deepchecks.nlp.checks import TextDuplicates
from deepchecks.nlp import TextData

texts = [
    "Deep learning is a subset of machine learning.",
    "Deep learning is a subset of machine learning.",
    "Deep learning is a sub-set of Machine Learning.",
    "Deep learning is subset of machine learning",
    "Natural language processing is a subfield of AI.",
    "This is a unique text sample.",
    "This is another unique text.",
]

dataset = TextData(texts)

#%%
# Run the Check
# =============

# Run the check without any text normalization
TextDuplicates(
    ignore_case=False,
    remove_punctuation=False,
    normalize_unicode=False,
    remove_stopwords=False,
    ignore_whitespace=False
).run(dataset)

# %%
# With Text Normalization
# -----------------------
# By default, ``TextDuplicates`` check applies text normalization before identifying the duplicates.
# This includes case normalization, punctuation removal, Unicode normalization and stopwords removal.
# You can also customize the normalization as per your requirements:

TextDuplicates(
    ignore_case=True,
    remove_punctuation=True,
    normalize_unicode=True,
    remove_stopwords=True,
    ignore_whitespace=True
).run(dataset)

# %%
# Of all the parameters in this example, ``ignore_whitespace`` is the only one set to ``False`` by default.
#
# Define a Condition
# ==================
#
# Now, we define a condition that enforces the ratio of duplicates to be 0. A condition
# is deepchecks' way to validate model and data quality, and let you know if anything
# goes wrong.

check = TextDuplicates()
check.add_condition_ratio_less_or_equal(0)
result = check.run(dataset)
result.show(show_additional_outputs=False)
