# -*- coding: utf-8 -*-
"""
Simple Model Comparison
***********************

This notebooks provides an overview for using and understanding simple model comparison check.

**Structure:**

* `What Is the Purpose of the Check? <#what-is-the-purpose-of-the-check>`__
* `Generate data an model <#generate-data-and-model>`__
* `Run the check <#run-the-check>`__

What Is the Purpose of the Check?
=================================
This check compares your current model to a "simple model", which is a model designed to produce the best performance
achievable using very simple rules, such as "always predict the most common class".
The simple model is used as a **baseline** model; If your model achieves less or similar score to the simple model,
this is an indicator of a possible problem with the model (e.g. it wasn't trained properly).

Using the parameter ``strategy``, you can select the simple model used in the check:

================  ===================================
Strategy          Description
================  ===================================
prior (default)   The probability vector always contains the empirical class prior distribution (i.e. the class
                  distribution observed in the training set).
most_frequent     The most frequent prediction is predicted. The probability vector is 1 for the most frequent
                  prediction and 0 for the other predictions.
stratified        The predictions are generated by sampling one-hot vectors from a multinomial distribution
                  parametrized by the empirical class prior probabilities.
uniform           Generates predictions uniformly at random from the list of unique classes observed in y, i.e. each
                  class has equal probability.
================  ===================================

Similiar to the :doc:`tabular simple model comparison check
</checks_gallery/tabular/model_evaluation/plot_simple_model_comparison>`,
there is no simple model which is more "correct" to use, each gives a different baseline
to compare to, and you may experiment with the different types and see how it performs
on your data.

This checks applies only to classification datasets.
"""

#%%
# Generate data and model
# -----------------------

from deepchecks.vision.checks import SimpleModelComparison
from deepchecks.vision.datasets.classification import mnist

#%%


mnist_model = mnist.load_model()
train_ds = mnist.load_dataset(train=True, object_type='VisionData')
test_ds = mnist.load_dataset(train=False, object_type='VisionData')

#%%
# Run the check
# -------------
# We will run the check with the prior model type. The check will use the default
# classification metrics - precision and recall. This can be overridden by
# providing an alternative scorer using the ``alternative_metrics``` parameter.

check = SimpleModelComparison(strategy='stratified')
result = check.run(train_ds, test_ds, mnist_model)

#%%
result

#%%
# Observe the check's output
# --------------------------
# We can see in the results that the check calculates the score for each class
# in the dataset, and compares the scores between our model and the simple model.
#
# In addition to the graphic output, the check also returns a value which includes
# all of the information that is needed for defining the conditions for validation.
#
# The value is a dataframe that contains the metrics' values for each class and dataset:

result.value.sort_values(by=['Class', 'Metric']).head(10)

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

check = SimpleModelComparison(strategy='stratified')
check.add_condition_gain_greater_than(min_allowed_gain=0.99)
result = check.run(train_ds, test_ds, mnist_model)
result

#%%
# We detected that for several classes our gain did not passed the target gain we
# defined, therefore it failed.
