# -*- coding: utf-8 -*-
"""
.. _plot_vision_single_dataset_performance:

Single Dataset Performance
*********************************

This notebooks provides an overview for using and understanding single dataset performance check.

**Structure:**

* `What Is the Purpose of the Check? <#what-is-the-purpose-of-the-check>`__
* `Generate Dataset <#generate-dataset>`__
* `Run the check <#run-the-check>`__
* `Define a condition <#define-a-condition>`__

What Is the Purpose of the Check?
=================================
This check returns the results from a dict of metrics, in the format metric name: scorer, calculated for the
given model dataset. The scorer should be either a sklearn scorer or a custom metric (see
:doc:`Metrics Guide </user-guide/general/metrics_guide>` for further details).
Use this check to evaluate the performance on a single vision dataset such as a test set.

"""
#%%
# Generate Dataset
# ----------------
#
# .. note::
#   In this example, we use the pytorch version of the mnist dataset and model. In order to run this example using
#   tensorflow, please change the import statements to::
#
#       from deepchecks.vision.datasets.classification import mnist_tensorflow as mnist

from deepchecks.vision.checks import SingleDatasetPerformance
from deepchecks.vision.datasets.classification import mnist_torch as mnist

#%%

train_ds = mnist.load_dataset(train=True, object_type='VisionData')

#%%
# Run the check
# -------------
#
# The check will use the default classification metrics - precision and recall.


check = SingleDatasetPerformance()
result = check.run(train_ds)
result.show()

#%%
# To display the results in an IDE like PyCharm, you can use the following code:

#  result.show_in_window()
#%%
# The result will be displayed in a new window.

#%%
# Now we will run a check with a metric different from the defaults- F-1.
# You can read more about setting metrics in the :doc:`Metrics Guide </user-guide/general/metrics_guide>`.

check = SingleDatasetPerformance(scorers={'f1': 'f1_per_class'})
result = check.run(train_ds)
result

#%%
# Define a Condition
# ==================
# We can define a condition to validate that our model performance score is above or below a certain threshold.
# The condition is defined as a function that takes the results of the check as input and
# returns a ConditionResult object.

check = SingleDatasetPerformance()
check.add_condition_greater_than(0.5)
result = check.run(train_ds)
result.show(show_additional_outputs=False)

#%%
# We can also define a condition on a specific metric (or a subset of the metrics) that was passed to the check and a
# specific class, instead of testing all the metrics and all the classes which is the default mode.

check = SingleDatasetPerformance()
check.add_condition_greater_than(0.8, metrics=['Precision'], class_mode='3')
result = check.run(train_ds)
result.show(show_additional_outputs=False)
