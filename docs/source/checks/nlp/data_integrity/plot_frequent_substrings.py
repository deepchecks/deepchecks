# -*- coding: utf-8 -*-
"""
.. _nlp__frequent_substrings:

Frequent Substrings
********************

This notebook provides an overview for using and understanding the frequent substrings check:

**Structure:**

* `Why check for frequent substrings? <#why-check-for-frequent-substrings>`__
* `Create TextData <#create-textdata>`__
* `Run the Check <#run-the-check>`__
* `Define a Condition <#define-a-condition>`__

Why check for frequent substrings?
===================================

The purpose of the ``FrequentSubstrings`` check is to identify recurring substrings within the Dataset.
These commonly occurring substrings can signal potential issues within the data pipeline that demand consideration.
Furthermore, these substrings might impact the model's performance and,
in certain scenarios, it might be necessary to remove them from the dataset.

Substrings of varying lengths (n-grams) are extracted from the dataset text samples.
The frequencies of these n-grams are calculated and only substrings exceeding a defined minimum length are retained.
The substrings are then sorted by their frequencies and the most frequent substrings are identified.
Finally, the substrings with the highest frequency and those surpassing a significance level are displayed.

Create TextData
===============

Let's create a simple dataset with some frequent substrings.
"""

from deepchecks.nlp import TextData
from deepchecks.nlp.checks import FrequentSubstrings

texts = [
    "Deep learning is a subset of machine learning. Sent from my iPhone",
    "Deep learning is a sub-set of Machine Learning.",
    "Natural language processing is a subfield of AI. Sent from my iPhone",
    "NLP is a subfield of Artificial Intelligence. Sent from my iPhone",
    "This is a unique text sample.",
    "This is another unique text.",
]

dataset = TextData(texts)

# %%
# Run the Check
# =============

FrequentSubstrings().run(dataset)

# %%
# Define a Condition
# ==================
#
# Now, we define a condition that enforces that ratio of frequent substrings will be smaller than 0.05
# for all frequent substrings in the data. A condition is deepchecks' way to validate model and data quality,
# and let you know if anything goes wrong.

check = FrequentSubstrings()
check.add_condition_zero_result()
result = check.run(dataset)
result.show(show_additional_outputs=False)
