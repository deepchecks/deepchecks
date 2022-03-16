# -*- coding: utf-8 -*-
"""
Trust Score Comparison
**********************
This notebooks provides an overview for using and understanding the trust score
comparison check.

**Structure:**

* `What is trust score? <#what-is-trust-score>`__
* `Loading the data <#loading-the-data>`__
* `Run the check <#run-the-check>`__
* `Define a condition <#define-a-condition>`__

What is trust score?
====================
Trust score is an alternative measure of model confidence, used in classification
problems to assign a higher score to samples whose prediction is more likely
to end up correct.

What is model confidence
^^^^^^^^^^^^^^^^^^^^^^^^
Model confidence commonly refers to the predicted probability of classification
model for the predicted class. This quantity is useful for a variety of tasks:

1. Detecting "problematic samples" before labels become available - predictions
   with low probability are more likely to be wrong.
2. Risk management - in use-cases such as loan approval, we may want to weigh
   the probability that the loan will be returned with the loaned sum and the
   expected return.
3. Early warning of concept drift - a significant decline in the average confidence
   of samples encountered in production or test data indicates that the model
   is predicting on more and more samples on which it is unsure.

Trust Score compared to predicted probability
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
"Regular" model confidence is easy to compute - just use the model's "predict_proba"
function. The danger with relying on the values produced by the model itself is
that they are often un-calibrated - which means that predicted probabilities
don't correspond to the actual percent of correct predictions (check the
`calibration score
</examples/tabular/checks/performance/test_autoexamples/plot_calibration_score.html>`__
). This is because the methods and loss functions used by these models are
often not designed to produce actual probabilities. Additionally, most common
classification metrics (such as precision, recall, accuracy etc.) measure only
the quality of the final prediction (after threshold is applied to the predicted
probability) and not on the probability itself. This reinforces the tendency to
ignore the quality of the probabilities themselves.

Trust Score is an alternative method for scoring the "trust-worthiness" of
the model predictions that is completely independent of model implementation.
The method and code used by the deepchecks package were published in
`To Trust Or Not To Trust A Classifier <https://arxiv.org/abs/1805.11783>`__.

Trust score has been shown to perform better than predicted probability in identifying
correctly classified samples, and is used by the TrustScoreComparison check for:

1. Identifying the samples with highest (and lowest) score - which are the
   samples most likely (and unlikely) to be correctly classified by the model.
   This is useful for visually detecting common qualities among the highest
   and lowest confidence samples.
2. Identifying a degradation between the trust score on the test data when
   comparing it to the training data, which may indicate that the model will
   perform worse on test compared to train and serves as a method to detect
   concept drift. This condition is useful especially for cases when the test
   labels are not available, such as when performing inference on new and unknown data.
"""

#%%
# Loading the data
# ================
# We'll load the scikit-learn breast cancer dataset to test out the Trust Score check.

import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from deepchecks.tabular.datasets.classification.breast_cancer import load_data
from deepchecks.tabular import Dataset

label = 'target'

train_df, test_df = load_data(data_format='Dataframe')
train = Dataset(train_df, label=label)
test = Dataset(test_df, label=label)

clf = AdaBoostClassifier()
features = train_df.drop(label, axis=1)
target = train_df[label]
clf = clf.fit(features, target)

#%%
# Run the check
# =============
# Next, we'll run the check on the dataset and model, modifying the default value of
# ``min_test_samples`` in order to enable us to run this check on the small dataset.
# In this case, we'll run the check "as is", and introduce the condition in the `next
# section <#define-a-condition>`__.
#
# Additional optional parameters include the maximal sample size, the random state,
# the number of highest and lowest Trust Score samples to show and various
# hyperparameters controlling the trust score algorithm.

from deepchecks.tabular.checks import TrustScoreComparison

TrustScoreComparison(min_test_samples=100).run(train, test, clf)

#%%
# Analyzing the output
# --------------------
# From here we can see that high trust score predictions are mostly correct, while the
# lowest trust score samples are wrong more often than not and are always predicted
# to belong to the negative class.
#
# Furthermore, we may notice some other common characteristics, such as the fact that
# ``worst texture`` and ``mean texture`` both seem to be lower in the top scoring samples,
# while the worst scoring samples have high ``worst texture`` and ``mean texture``
# values, both features with high feature importance for the AdaBoost model. Might
# it be that high texture samples are getting worse predictions by the model?

pd.Series(index=train_df.columns[:-1] ,data=clf.feature_importances_, name='Model Feature importance').sort_values(ascending=False).to_frame().head(7)

#%%
# Define a condition
# ==================
# Introducing concept drift
# -------------------------
# First, we introduce concept drift into the data by changing the relation between the
# worst texture and mean concave points features, both important features for the model.

mod_test_df = test_df.copy()
np.random.seed(0)
sample_idx = np.random.choice(test_df.index, 80, replace=False)
mod_test_df.loc[sample_idx, 'worst texture'] = mod_test_df.loc[sample_idx, 'target'] * (mod_test_df.loc[sample_idx, 'mean concave points'] > 0.05)
mod_test = Dataset(mod_test_df, label=label)

#%%
# Checking for decline in Trust Score
# -----------------------------------
# Now, we define a condition on the Trust Score check to alert us on significant
# degradation in the mean Trust Score of the test data compared to the training data.
# Note that the threshold percent of decline can be modified by passing a different
# threshold to the condition (the default is 0.2, or 20% decline).

from deepchecks.tabular.checks import TrustScoreComparison

TrustScoreComparison(min_test_samples=100).add_condition_mean_score_percent_decline_not_greater_than(threshold=0.19).run(train, mod_test, clf)

#%%
# Analyzing the output
# --------------------
# The condition alerts us to the fact that the mean Trust Score has declined by ~21%,
# which is more than the 10% we allowed!
#
# The decline is also evident in the plot showing the distribution of Trust Scores
# in each dataset, in which we can see that test data has significantly more samples
# with Trust Score around 1 compared to training data. We can also see the distribution
# of the Trust Score for the modified test data used here is visibly skewed to the left
# (low Trust Score) due to the introduction of concept drift into the test data. The
# condition helps us detect this new skew. Did this skew in the data really change
# the performance of the model?

from deepchecks.tabular.checks.performance import MultiModelPerformanceReport

#%%

MultiModelPerformanceReport().run([train, train], [test, mod_test], {'unmodified test': clf, 'modified test': clf})

#%%
# Using the MultiModelPerformanceReport we can clearly see that several metrics (such
# as f1, and recall) have declined on the modified test dataset. In a use case in which
# labels were not available for test data, we would have still known to be wary of
# that thanks to the condition raised by the Trust Score check on the modified data!
