# -*- coding: utf-8 -*-
"""
.. _plot_tabular_string_mismatch_comparison:

String Mismatch Comparison
**************************

This page provides an overview for using and understanding the "String Mismatch Comparison" check.

**Structure:**

* `What is the purpose of the check? <#what-is-the-purpose-of-the-check>`__
* `Run check <#run-the-check>`__
* `Define a condition <#define-a-condition>`__


What is the purpose of the check?
=================================
The check compares the same categorical column within train and test and checks whether there are variants of similar
strings that exists only in test and not in train.
Finding those mismatches is helpful to prevent errors when inferring on the test data. For example, in train data we
have category 'New York', and  in our test data we have 'new york'. We would like to be acknowledged that the test
data contain a new variant of the train data, so we can address the problem.

How String Mismatch Defined?
----------------------------
To recognize string mismatch, we transform each string to it's base form. The base form is the string with only its
alphanumeric characters in lowercase. (For example "Cat-9?!" base form is "cat9"). If two strings have the same base
form, they are considered to be the same.
"""

import pandas as pd

#%%
# Run the Check
# =============
from deepchecks.tabular.checks import StringMismatchComparison

data = {'col1': ['Deep', 'deep', 'deep!!!', 'earth', 'foo', 'bar', 'foo?']}
compared_data = {'col1': ['Deep', 'deep', '$deeP$', 'earth', 'foo', 'bar', 'foo?', '?deep']}

check = StringMismatchComparison()
result = check.run(pd.DataFrame(data=data), pd.DataFrame(data=compared_data))
result

#%%
# Define a Condition
# ==================

check = StringMismatchComparison().add_condition_no_new_variants()
result = check.run(pd.DataFrame(data=data), pd.DataFrame(data=compared_data))
result.show(show_additional_outputs=False)
