# -*- coding: utf-8 -*-
"""
.. _plot_tabular_string_length_out_of_bounds:

String Length Out Of Bounds
***************************

This notebook provides an overview for using and understanding the String Length Out Of Bounds check.

**Structure:**

* `What is String Length Out Of Bounds <#what-is-string-length-out-of-bounds>`__
* `Generate data <#generate-data>`__
* `Run the check <#run-the-check>`__
"""

# %%
# What is String Length Out Of Bounds
# ======================================
# The ``StringLengthOutOfBounds`` check detects strings with length that is much longer/shorter
# than the identified "normal" string lengths.

#%%

import pandas as pd

from deepchecks.tabular.checks import StringLengthOutOfBounds
from deepchecks.tabular import Dataset
#%%
# Generate Data
# ===============
col1 = ["deepchecks123", "deepchecks456"]*40
col1.append("ab")
col1.append("cd")

col2 = ["b", "abc"]*41

col3 = ["deepchecks"]*80
col3.append("an_outlier")
col3.append("im_an_outlier_too")

## col1 and col3 contain outliers, col2 does not
df = pd.DataFrame({"col1":col1, "col2": col2, "col3": col3 })
df = Dataset(df, cat_features=[])

#%%
# Run the check
# ================
StringLengthOutOfBounds().run(df)


