# -*- coding: utf-8 -*-
"""

.. _nlp__unknown_tokens:

Unknown Tokens
**************

This notebook provides an overview for using and understanding the Unknown Tokens check.

**Structure:**

* `What is the purpose of the check? <#what-is-the-purpose-of-the-check>`__
* `Generate data & model <#generate-data-model>`__
* `Run the check <#run-the-check>`__
* `Using the Check Value <#using-the-check-value>`__
* `Define a condition <#define-a-condition>`__

What is the purpose of the check?
==================================

The Unknown Tokens check is designed to help you identify samples that contain tokens not supported by your tokenizer.
These not supported tokens can lead to poor model performance, as the model may not be able to understand the meaning
of such tokens. By identifying these unknown tokens, you can take appropriate action, such as updating your tokenizer
or preprocessing your data to handle them.

Generate data & model
=====================

In this example, we'll use the twitter dataset.

"""

from deepchecks.nlp.datasets.classification import tweet_emotion

dataset, _ = tweet_emotion.load_data()

# %%
# Run the check
# =============
#
# The check has several key parameters that affect its behavior and output:
#
# * `tokenizer`: Tokenizer from the HuggingFace transformers library to use for tokenization. If None,
#   BertTokenizer.from_pretrained('bert-base-uncased') will be used.
# * `group_singleton_words`: If True, group all words that appear only once in the data into the "Other" category in
#   the display.


from deepchecks.nlp.checks import UnknownTokens

check = UnknownTokens()
result = check.run(dataset)
result.show()

# %%
# Observe the check's output
# --------------------------
#
# We see in the results that the check found many emojis and some foreign words (Korean, can be seen by hovering
# over the "Other Unknown Words" slice of the pie chart) that are not supported by the
# tokenizer. We can also see that the check grouped all words that appear only once in the data into the "Other"
#
# Use a Different Tokenizer
# -------------------------
#
# We can also use a different tokenizer, such as the GPT2 tokenizer, to see how the results change.

from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

UnknownTokens(tokenizer=tokenizer).run(dataset)

# %%
# Using the Check Value
# =====================
#
# On top of observing the check's display, we can use the check's returned value to get more information about the
# words containing unknown tokens in our dataset. The check's value is a nested dictionary with the following keys:
#
# 1. ``unknown_word_ratio``: The ratio of unknown words out of all words in the dataset.
# 2. ``unknown_word_details``: This is in turn also a dict, containing a key for each unknown word. The value for each
#    key is a dict containing 'ratio' (the ratio of the unknown word out of all words in the dataset) and 'indexes'
#    (the indexes of the samples containing the unknown word).
#
# We'll show here how you can use this value to get the individual samples containing unknown tokens.

from pprint import pprint

unknown_word_details = result.value['unknown_word_details']
first_unknown_word = list(unknown_word_details.keys())[0]
print(f"Unknown word: {first_unknown_word}")

word_indexes = unknown_word_details[first_unknown_word]['indexes']
pprint(dataset.text[word_indexes].tolist())

# %%
#
# As we can see, the GPT2 tokenizer supports emojis, so the check did not find any unknown tokens.
#
# Define a condition
# ==================
#
# We can add a condition that validates the ratio of unknown words in the dataset is below a certain threshold. This can
# be useful to ensure that your dataset does not have a high percentage of unknown tokens, which might negatively impact
# the performance of your model.

check.add_condition_ratio_of_unknown_words_less_or_equal(0.005)
result = check.run(dataset)
result.show(show_additional_outputs=False)

# %%
# In this example, the condition checks if the ratio of unknown words is less than or equal to 0.005 (0.5%). If the
# ratio is higher than the threshold, the condition will fail, indicating a potential issue with the dataset.
