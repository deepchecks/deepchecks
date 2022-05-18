# -*- coding: utf-8 -*-
"""
Single Dataset Scalar Performance
*********************************

This notebooks provides an overview for using and understanding single dataset scalar performance check.

**Structure:**

* `What Is the Purpose of the Check? <#what-is-the-purpose-of-the-check>`__
* `Generate data an model <#generate-data-and-model>`__
* `Run the check <#run-the-check>`__
* `Define a condition <#define-a-condition>`__

What Is the Purpose of the Check?
=================================
This check returns a metric result as a single scalar, which is especially useful for monitoring a model in production.

Some metrics return a single score, but others return a tensor of scores.

For example, Precision returns a tensor in the size of the number of classes. In that case, we will use a
reduce function - a function that aggregates the scores into a scalar.
In this example we use 'nanmean' that returns the mean over the classes, while ignoring NaNs.

"""
#%%
# Generate data and model
# -----------------------

from deepchecks.vision.checks.performance.single_dataset_scalar_performance import \
    SingleDatasetScalarPerformance
from deepchecks.vision.datasets.classification import mnist

#%%


mnist_model = mnist.load_model()
train_ds = mnist.load_dataset(train=True, object_type='VisionData')

#%%
# Run the check
# -------------
# We will run the check with the model defined above.
#
# The check will use the default classification metric -
# `ignite.Accuracy <https://pytorch.org/ignite/generated/ignite.metrics.Accuracy.html>`__.
# The default metric returns a scalar, therefore we will use the reduce function default - None.


check = SingleDatasetScalarPerformance()
result = check.run(train_ds, mnist_model)
result

#%%
# The result value is a dictionary with the following fields:
# score - the actual result,
# metric - the name of metric used
# reduce - the name of the reduce function used.
result.value

#%%
# Now we will run a check with parameters, to use a metric and a reduce function different from the defaults.
# We will also pass names for them, so that the return value will look neat.
from ignite.metrics import Precision
from torch import nanmean

check = SingleDatasetScalarPerformance(Precision(), nanmean, metric_name='precision', reduce_name='mean')
result = check.run(train_ds, mnist_model)
result.value

#%%
# Define a Condition
# ==================
# We can define a condition to validate that our model performance score is above or below a certain threshold.
# The condition is defined as a function that takes the results of the check as input and
# returns a ConditionResult object.

check = SingleDatasetScalarPerformance()
check.add_condition_greater_than(0.5)
result = check.run(train_ds, mnist_model)
result.show(show_additional_outputs=False)


