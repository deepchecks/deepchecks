# -*- coding: utf-8 -*-
"""
.. _plot_tabular_conflicting_labels:

Conflicting Labels
******************

This notebook provides an overview for using and understanding the conflicting labels check.

**Structure:**

* `What are Conflicting Labels? <#what-are-conflicting-labels>`__
* `Load Data <#load-data>`__
* `Run the Check <#run-the-check>`__
* `Define a Condition <#define-a-condition>`__

What are Conflicting Labels?
============================
The check searches for identical samples with different labels. This can
occur due to either mislabeled data, or when the data collected is missing
features necessary to separate the labels. If the data is mislabled, it can
confuse the model and can result in lower performance of the model.
"""
import pandas as pd

from deepchecks.tabular import Dataset
# %%
from deepchecks.tabular.checks import ConflictingLabels
from deepchecks.tabular.datasets.classification.phishing import load_data

#%%
# Load Data
# =========


phishing_dataframe = load_data(as_train_test=False, data_format='Dataframe')
phishing_dataset = Dataset(phishing_dataframe, label='target', features=['urlLength', 'numDigits', 'numParams', 'num_%20', 'num_@', 'bodyLength', 'numTitles', 'numImages', 'numLinks', 'specialChars'])

#%%
# Run the Check
# =============

ConflictingLabels().run(phishing_dataset)

#%%
# We can also check label ambiguity on a subset of the features:

ConflictingLabels(n_to_show=1).run(phishing_dataset)

#%%

ConflictingLabels(columns=['urlLength', 'numDigits']).run(phishing_dataset)

#%%
# Define a Condition
# ==================
# Now, we define a condition that enforces that the ratio of samples with conflicting labels
# should be 0. A condition is deepchecks' way to validate model and data quality,
# and let you know if anything goes wrong.

check = ConflictingLabels()
check.add_condition_ratio_of_conflicting_labels_less_or_equal(0)
result = check.run(phishing_dataset)
result.show(show_additional_outputs=False)
