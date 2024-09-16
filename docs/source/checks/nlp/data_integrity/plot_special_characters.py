# -*- coding: utf-8 -*-
"""
.. _nlp__special_characters:

Special Characters
******************

This notebook provides an overview for using and understanding the special characters check.

**Structure:**

* `Why check for special characters? <#why-check-for-text-data-duplicates>`__
* `Generate data & model <#generate-data-model>`__
* `Run the Check <#run-the-check>`__
* `Define a Condition <#define-a-condition>`__

Why check for special characters?
===================================

The ``SpecialCharacters`` check looks for text sample in which the percentage of special characters
out of all characters is significant. Such samples can be an indicator for a problem in the data pipeline that
require attention. Additionally, such examples may be problematic for the model to predict on.
For example, a text sample with many emojis may be hard to
predict on and a common methodology will be to replace them with a textual representation of the emoji.

Generate data & model
=====================

Let's create a simple dataset with some duplicate and similar text samples.
"""

from deepchecks.nlp.datasets.classification import tweet_emotion

text_data = tweet_emotion.load_data(as_train_test=False)
text_data.head(3)

# %%
# Run the Check
# =============

from deepchecks.nlp.checks import SpecialCharacters

check = SpecialCharacters()
result = check.run(text_data)
result.show()

# %%
# We can see in the check display that ~17% of the samples contain at least one special character and that the
# samples with the highest percentage of special characters contain many emojis.
#
# In addition to the check display we can also see receive a summary of most common special characters
# and which samples contain them. This can assist us in conforming that the majority of the special characters
# in this dataset are indeed emojis.

result.value["samples_per_special_char"]

# %%
# Define a condition
# ==================
#
# We can add a condition that will validate that the percentage of samples with a significant ratio of
# special characters is below a certain threshold.
# Let's add a condition and re-run the check:
#

check.add_condition_samples_ratio_w_special_characters_less_or_equal(0.01)
result = check.run(text_data)
result.show()
