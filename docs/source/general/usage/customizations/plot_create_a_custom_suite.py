# -*- coding: utf-8 -*-
"""
Create a Custom Suite
*********************

A suite is a list of checks that will run one after the other, and its results will be
displayed together.

To customize a suite, we can either:

* `Create new custom suites <#create-a-new-suite>`__, by choosing the checks (and
  the optional conditions) that we want the suite to have.
* `Modify a built-in suite <#modify-an-existing-suite>`__ by adding and/or removing
  checks and conditions, to adapt it to our needs.

Create a New Suite
==================
Let's say we want to create our custom suite, mainly with various performance checks,
including ``PerformanceReport(), TrainTestDifferenceOverfit()`` and several more.

For assistance in understanding which checks are implemented and can be included,
we suggest using any of:

* :doc:`API Reference </api/index>`
* `Tabular checks demonstration notebooks </examples/index.html#tabular-examples>`__
* `Computer vision checks demonstration notebooks </examples/index.html#computer-vision-examples>`__
* Built-in suites (by printing them to see which checks they include)
"""

#%%

from sklearn.metrics import make_scorer, precision_score, recall_score

from deepchecks.tabular import Suite
# importing all existing checks for demonstration simplicity
from deepchecks.tabular.checks import *

# The Suite's first argument is its name, and then all of the check objects.
# Some checks can receive arguments when initialized (all check arguments have default values)
# Each check can have an optional condition(/s)
# Multiple conditions can be applied subsequentially
new_custom_suite = Suite('Simple Suite For Model Performance',
                         ModelInfo(),
                         # use custom scorers for performance report:
                         TrainTestPerformance().add_condition_train_test_relative_degradation_less_than(threshold=0.15)\
                         .add_condition_test_performance_greater_than(0.8),
                         ConfusionMatrixReport(),
                         SimpleModelComparison(strategy='most_frequent',
                                               scorers={'Recall (Multiclass)': make_scorer(recall_score, average=None),
                                                        'Precision (Multiclass)': make_scorer(precision_score, average=None)}
                                               ).add_condition_gain_greater_than(0.3)
                         )

# The scorers parameter can also be passed to the suite in order to override the scorers of all the checks in the suite.
# Find more about scorers at https://docs.deepchecks.com/stable/user-guide/general/metrics_guide.html.

#%%
# Let's see the suite:
new_custom_suite

#%%
# *TIP: the auto-complete may not work from inside a new suite definition, so if you want
# to use the auto-complete to see the arguments a check receive or the built-in conditions
# it has, try doing it outside of the suite's initialization.*
#
# *For example, to see a check's built-in conditions, type in a new cell:
# ``NameOfDesiredCheck().add_condition_`` and then check the auto-complete suggestions
# (using Shift + Tab), to discover the built-in checks.*
#
# Additional Notes about Conditions in a Suite
# --------------------------------------------
# * Checks in the built-in suites come with pre-defined conditions, and when building
#   your custom suite you should choose which conditions to add.
# * Most check classes have built-in methods for adding conditions. These apply to the
#   naming convention ``add_condition_...``, which enables adding a condition logic to parse
#   the check's results.
# * Each check instance can have several conditions or none. Each condition will be
#   evaluated separately.
# * The pass (✓) / fail (✖) / insight (!) status of the conditions, along with the
#   condition's name and extra info will be displayed in the suite's Conditions Summary.
# * Most conditions have configurable arguments that can be passed to the condition while adding it.
# * For more info about conditions, check out :doc:`Configure a Condition
#   <plot_configure_check_conditions>`.
#
# Run the Suite
# =============
# This is simply done by calling the ``run()`` method of the suite.
#
# To see that in action, we'll need datasets and a model.
#
# Let's quickly load a dataset and train a simple model for the sake of this demo
#
# Load Datasets and Train a Simple Model
# --------------------------------------

import numpy as np
# General imports
import pandas as pd

np.random.seed(22)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from deepchecks.tabular.datasets.classification import iris

# Load pre-split Datasets
train_dataset, test_dataset = iris.load_data(as_train_test=True)
label_col = 'target'

# Train Model
rf_clf = RandomForestClassifier()
rf_clf.fit(train_dataset.data[train_dataset.features],
           train_dataset.data[train_dataset.label_name]);

#%%
# Run Suite
# ---------

new_custom_suite.run(model=rf_clf, train_dataset=train_dataset, test_dataset=test_dataset)

#%%
# Modify an Existing Suite
# ========================

from deepchecks.tabular.suites import train_test_validation

customized_suite = train_test_validation()

# let's check what it has:
customized_suite

#%%

# and modify it by removing a check by index:
customized_suite.remove(1)

#%%

from deepchecks.tabular.checks import UnusedFeatures

# and add a new check with a condition:
customized_suite.add(
    UnusedFeatures().add_condition_number_of_high_variance_unused_features_less_or_equal())

#%%

# lets remove all condition for the FeatureLabelCorrelationChange:
customized_suite[3].clean_conditions()

# and update the suite's name:
customized_suite.name = 'New Data Leakage Suite'

#%%

# and now we can run our modified suite:
customized_suite.run(train_dataset, test_dataset, rf_clf)
