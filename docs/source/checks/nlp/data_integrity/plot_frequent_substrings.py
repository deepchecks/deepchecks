# -*- coding: utf-8 -*-
"""
.. _nlp__frequent_substrings:

Frequent Substrings
********************

This notebook provides an overview for using and understanding the frequent substrings check:

**Structure:**

* `Why check for frequent substrings? <#why-check-for-text-data-duplicates>`__
* `Create TextData <#create-textdata>`__
* `Run the Check <#run-the-check>`__
* `Define a Condition <#define-a-condition>`__

Why check for frequent substrings?
===================================

The ``FrequentSubstrings`` check finds frequent substrings of varying lengths (n-grams) in the Dataset.
Frequent substrings may be an indicator for a problem in the data pipeline that requires attention.

Create TextData
===============

Let's create a simple dataset with some frequent substrings.
"""

from deepchecks.nlp.checks import FrequentSubstrings
from deepchecks.nlp import TextData

texts = [
    'Deep learning is a subset of machine learning. Sent from my iPhone',
    'Deep learning is a sub-set of Machine Learning.',
    'Natural language processing is a subfield of AI. Sent from my iPhone',
    'NLP is a subfield of Artificial Intelligence. Sent from my iPhone',
    'This is a unique text sample.',
    'This is another unique text.'
]

dataset = TextData(texts)

#%%
# Run the Check
# =============

FrequentSubstrings().run(dataset)

# %%
# Define a Condition
# ==================
#
# Now, we define a condition that enforces the ratio of frequent substrings to be smaller than 0.05
# for all frequent substrings in the data. A condition is deepchecks' way to validate model and data quality,
# and let you know if anything goes wrong.

check = FrequentSubstrings()
check.add_condition_zero_result()
result = check.run(dataset)
result.show(show_additional_outputs=False)
