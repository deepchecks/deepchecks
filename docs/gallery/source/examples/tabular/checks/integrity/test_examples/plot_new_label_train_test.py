# -*- coding: utf-8 -*-
"""
New Label
*********
"""

#%%

from deepchecks.tabular.checks.integrity.new_label import NewLabelTrainTest
from deepchecks.tabular.base import Dataset
import pandas as pd

#%%

test_data = {"col1": [0, 1, 2, 3] * 10}
val_data = {"col1": [4, 5, 6, 7, 8, 9] * 10}
test = Dataset(pd.DataFrame(data=test_data), label="col1", label_type="classification_label")
val = Dataset(pd.DataFrame(data=val_data), label="col1", label_type="classification_label")

#%%

test_data = {"col1": ["a", "b", "a", "c"] * 10, "col2": [1,2,2,3]*10}
val_data = {"col1": ["a","b","d"] * 10, "col2": [1, 4, 5]*10}
test = Dataset(pd.DataFrame(data=test_data), label="col2", label_type="classification_label")
val = Dataset(pd.DataFrame(data=val_data), label="col2", label_type="classification_label")

#%%
NewLabelTrainTest().run(test, val)
