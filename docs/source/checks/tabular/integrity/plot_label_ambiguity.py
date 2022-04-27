# -*- coding: utf-8 -*-
"""
Label Ambiguity
***************

This notebooks provides an overview for using and understanding the label ambiguity check.

**Structure:**

* `What is Label Ambiguity? <#what-is-label-ambiguity>`__
* `Load Data <#load-data>`__
* `Run the Check <#run-the-check>`__
* `Define a Condition <#define-a-condition>`__

What is Label Ambiguity?
========================
Label Ambiguity searches for identical samples with different labels. This can
occur due to either mislabeled data, or when the data collected is missing
features necessary to separate the labels. If the data is mislabled, it can
confuse the model and can result in lower performance of the model.
"""
# %%
from deepchecks.tabular.checks.integrity import LabelAmbiguity
from deepchecks.tabular import Dataset
import pandas as pd

#%%
# Load Data
# =========

from deepchecks.tabular.datasets.classification.phishing import load_data

phishing_dataframe = load_data(as_train_test=False, data_format='Dataframe')
phishing_dataset = Dataset(phishing_dataframe, label='target', features=['urlLength', 'numDigits', 'numParams', 'num_%20', 'num_@', 'bodyLength', 'numTitles', 'numImages', 'numLinks', 'specialChars'])

#%%
# Run the Check
# =============

LabelAmbiguity().run(phishing_dataset)

#%%
# We can also check label ambiguity on a subset of the features:

LabelAmbiguity(n_to_show=1).run(phishing_dataset)

#%%

LabelAmbiguity(columns=['urlLength', 'numDigits']).run(phishing_dataset)

#%%
# Define a condition
# ==================
# Now, we define a condition that enforces that the ratio of ambiguous samples
# should be 0. A condition is deepchecks' way to validate model and data quality,
# and let you know if anything goes wrong.

check = LabelAmbiguity()
check.add_condition_ambiguous_sample_ratio_not_greater_than(0)
result = check.run(phishing_dataset)
result.show(show_additional_outputs=False)
