# -*- coding: utf-8 -*-
"""
.. _plot_tabular_new_label:

New Label
*********
"""

#%%

import pandas as pd

from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import NewLabelTrainTest

#%%

test_data = {"col1": [0, 1, 2, 3] * 10}
val_data = {"col1": [4, 5, 6, 7, 8, 9] * 10}
test = Dataset(pd.DataFrame(data=test_data), label="col1", label_type="multiclass")
val = Dataset(pd.DataFrame(data=val_data), label="col1", label_type="multiclass")

#%%

test_data = {"col1": ["a", "b", "a", "c"] * 10, "col2": [1,2,2,3]*10}
val_data = {"col1": ["a","b","d"] * 10, "col2": [1, 4, 5]*10}
test = Dataset(pd.DataFrame(data=test_data), label="col2", label_type="multiclass")
val = Dataset(pd.DataFrame(data=val_data), label="col2", label_type="multiclass")

#%%
NewLabelTrainTest().run(test, val)
