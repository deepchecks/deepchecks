# -*- coding: utf-8 -*-
"""
Configure Check Conditions
**************************

The following guide includes different options for configuring a check's condition(s):

* `Add Condition <#add-condition>`__
* `Remove / Edit a Condition <#remove-edit-a-condition>`__
* `Add a Custom Condition <#add-a-custom-condition>`__
* `Set Custom Condition Category <#set-custom-condition-category>`__

Add Condition
=============
In order to add a condition to an existing check, we can use any of the pre-defined
conditions for that check. The naming convention for the methods that add the
condition is ``add_condition_...``.

If you want to create and add your custom condition logic for parsing the check's
result value, see `Add a Custom Condition <#add-a-custom-condition>`__.
"""

#%%
# Add a condition to a new check
# ------------------------------

from deepchecks.tabular.checks import DatasetsSizeComparison

check = DatasetsSizeComparison().add_condition_test_size_greater_or_equal(1000)
check

#%%
# Conditions are used mainly in the context of a Suite, and displayed in the
# Conditions Summary table. For example how to run in a suite you can look at
# `Add a Custom Condition <#add-a-custom-condition>`__ or if you would like to
# run the conditions outside of suite you can execute:

import pandas as pd

from deepchecks.tabular import Dataset

# Dummy data
train_dataset = Dataset(pd.DataFrame(data={'x': [1,2,3,4,5,6,7,8,9]}))
test_dataset = Dataset(pd.DataFrame(data={'x': [1,2,3]}))

condition_results = check.conditions_decision(check.run(train_dataset, test_dataset))
condition_results

#%%
# Add a condition to a check in a suite
# -------------------------------------
# If we want to add a conditon to a check within an existing suite, we should first
# find the Check's ID within the suite, and then add the condition to it, by running
# the relevant ``add_condition_`` method on that check's instance. See the next section
# to understand how to do so.
#
# The condition will then be appended to the list of conditions on that check (or be
# the first one if no conditions are defined), and each condition will be evaluated
# separately when running the suite.

#%%
# Remove / Edit a Condition
# =========================
# Deepchecks provides different kinds of default suites, which come with pre-defined
# conditions. You may want to remove a condition in case it isn't needed for you, or
# you may want to change the condition's parameters (since conditions functions are immutable).
#
# To remove a condition, start by printing the Suite and identifing the Check's ID,
# and the Condition's ID:

from deepchecks.tabular.suites import train_test_validation

suite = train_test_validation()
suite

#%%
# After we found the IDs we can remove the Condition:

# Access check by id
check = suite[8]
# Remove condition by id
check.remove_condition(0)

suite

#%%
# Now if we want we can also re-add the Condition using the built-in methods on the check,
# with a different parameter.

# Re-add the condition with new parameter
check.add_condition_feature_pps_difference_less_than(0.01)

suite

#%%
# Add a Custom Condition
# ======================
# In order to write conditions we first have to know what value a given check produces.
#
# Let's look at the check ``DatasetsSizeComparison`` and see it's return value in
# order to write a condition for it.

import pandas as pd

from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import DatasetsSizeComparison

# We'll use dummy data for the purpose of this demonstration
train_dataset = Dataset(pd.DataFrame(data={'x': [1,2,3,4,5,6,7,8,9]}))
test_dataset = Dataset(pd.DataFrame(data={'x': [1,2,3]}))

result = DatasetsSizeComparison().run(train_dataset, test_dataset)
result.value

#%%
# Now we know what the return value looks like. Let's add a new condition that validates
# that the ratio between the train and test datasets size is inside a given range. To
# create condition we need to use the ``add_condition`` method of the check which accepts
# a condition name and a function. This function receives the value of the ``CheckResult``
# that we saw above and should return either a boolean or a ``ConditionResult`` containing
# a boolean and optional extra info that will be displayed in the Conditions Summary table.
#
# *Note: When implementing a condition in a custom check, you may want to add a method
# ``add_condition_x()`` to allow any consumer of your check to apply the condition
# (possibly with given parameters). For examples look at implemented Checks' source code*

from deepchecks.core import ConditionResult

# Our parameters for the condition
low_threshold = 0.4
high_threshold = 0.6

# Create the condition function
def custom_condition(value: dict, low=low_threshold, high=high_threshold): 
    ratio = value['Test'] / value['Train']
    if low <= ratio <= high:
        return ConditionResult(ConditionCategory.PASS)
    else:
        # Note: if you doesn't care about the extra info, you can return directly a boolean
        return ConditionResult(ConditionCategory.FAIL, f'Test-Train ratio is {ratio:.2}')

# Create the condition name
condition_name = f'Test-Train ratio is between {low_threshold} to {high_threshold}'

# Create check instance with the condition 
check = DatasetsSizeComparison().add_condition(condition_name, custom_condition)

#%%
# Now we will use a Suite to demonstrate the action of the condition, since the suite
# runs the condition for us automatically and prints out a Conditions Summary table
# (for all the conditions defined on the checks within the suite):

from deepchecks.tabular import Suite

# Using suite to run check & condition
suite = Suite('Suite for Condition',
    check
)

suite.run(train_dataset, test_dataset)

#%%
# Set Custom Condition Category
# =============================
# When writing your own condition logic, you can decide to mark a condition result
# as either fail or warn, by passing the category to the ConditionResult object.
# For example we can even write condition which sets the category based on severity of the result:

from deepchecks.core import ConditionCategory, ConditionResult

# Our parameters for the condition
low_threshold = 0.3
high_threshold = 0.7

# Create the condition function for check `DatasetsSizeComparison`
def custom_condition(value: dict): 
    ratio = value['Test'] / value['Train']
    if low_threshold <= ratio <= high_threshold:
        return ConditionResult(ConditionCategory.PASS)
    elif ratio < low_threshold:
        return ConditionResult(ConditionCategory.FAIL, f'Test-Train ratio is {ratio:.2}', ConditionCategory.FAIL)
    else:
        return ConditionResult(ConditionCategory.FAIL, f'Test-Train ratio is {ratio:.2}', ConditionCategory.WARN)
