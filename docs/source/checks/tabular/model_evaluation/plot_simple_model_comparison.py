# -*- coding: utf-8 -*-
"""
.. _plot_tabular_simple_model_comparison:

Simple Model Comparison
***********************
This notebook provides an overview for using and understanding simple model comparison check.

**Structure:**

* `What is the purpose of the check? <#what-is-the-purpose-of-the-check>`__
* `Generate data & model <#generate-data-model>`__
* `Run the check <#run-the-check>`__
* `Define a condition <#define-a-condition>`__

What is the purpose of the check?
=================================
The simple model is designed to produce the best performance achievable using very
simple rules. The goal of the simple model is to provide a baseline of minimal model
performance for the given task, to which the user model may be compared. If the user
model achieves less or a similar score to the simple model, this is an indicator
for a possible problem with the model (e.g. it wasn't trained properly).

The check has three possible "simple model" heuristics, from which one is chosen and
compared to. By default the check uses the **most_frequent** heuristic, which can be
overridden in the checks' parameters using strategy. There is no simple
model which is more "correct" to use, each gives a different baseline to compare to,
and you may experiment with the different types and see how it performs on your data.

The simple models are:

* A **most_frequent** model - The default. In regression the prediction is equal to the
  mean value, in classification the prediction is equal to the most common value.
* A **uniform** model - In regression, selects a random value from the y range.
  In classification, selects one of the labels by random.
* A **stratified** model - Draws the prediction from the distribution of the labels in the train.
* A **tree** model - Trains a simple decision tree with a given depth. The depth
  can be customized using the ``max_depth`` parameter.
"""

#%%
# Generate data & model
# =====================

from deepchecks.tabular.datasets.classification.phishing import (
    load_data, load_fitted_model)

train_dataset, test_dataset = load_data()
model = load_fitted_model()

#%%
# Run the check
# =============
# We will run the check with the **tree** model type. The check will use the default
# metric defined in deepchecks for classification. This can be overridden by
# providing an alternative scorer using the ``alternative_scorers`` parameter.
#
# Note that we are demonstrating on a classification task, but the check also works
# for regression tasks. For classification we will see the metrics per class, and for
# regression we'll have a single result per metric.

from deepchecks.tabular.checks import SimpleModelComparison

# Using tree model as a simple model, and changing the tree depth from the default 3 to 5
check = SimpleModelComparison(strategy='tree', max_depth=5)
check.run(train_dataset, test_dataset, model)

#%%
# Observe the check's output
# --------------------------
# We can see in the results that the check calculates the score for each class in
# the dataset, and compares the scores between our model and the simple model.
# In addition to the graphic output, the check also returns a value which includes
# all of the information that is needed for defining the conditions for validation.
#
# The value is a dictionary of:
#
# - scores - for each metric and class returns the numeric score
# - type - the model task type
# - scorers_perfect - for each metric the perfect possible score (used to calculate gain)
# - classes - the classes exists in the data
#
# Note: for regression ``scores`` will contain for each metric a single numeric score,
# and ``classes`` will be null.

check = SimpleModelComparison()
result = check.run(train_dataset, test_dataset, model)
result.value

#%%
# Define a condition
# ==================
# We can define on our check a condition that will validate our model is better than
# the simple model by a given margin called gain. For classification we check the gain
# for each class separately and if there is a class that doesn't pass the defined gain
# the condition will fail.
#
# The performance gain is the percent of the improved performance out of the
# "remaining" unattained performance. Its purpose is to reflect the significance of
# the said improvement. Take for example for a metric between 0 and 1. A change of
# only 0.03 that takes us from 0.95 to 0.98 is highly significant (especially in an
# imbalance scenario), but improving from 0.1 to 0.13 is not a great achievement.
#
# The gain is calculated as: :math:`gain = \frac{\text{model score} - \text{simple score}}
# {\text{perfect score} - \text{simple score}}`
#
# Let's add a condition to the check and see what happens when it fails:

check = SimpleModelComparison(strategy='tree')
check.add_condition_gain_greater_than(0.9)
result = check.run(train_dataset, test_dataset, model)
result.show(show_additional_outputs=False)

#%%
# We detected that for class "1" our gain did not passed the target gain we
# defined, therefore it failed.
