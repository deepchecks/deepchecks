# -*- coding: utf-8 -*-
"""
.. _plot_tabular_index_leakage:

Index Leakage
*************
"""

#%%

import pandas as pd

from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import IndexTrainTestLeakage

#%%

def dataset_from_dict(d: dict, index_name: str = None) -> Dataset:
    dataframe = pd.DataFrame(data=d)
    return Dataset(dataframe, index_name=index_name)

#%%
# Synthetic example with index leakage
# ====================================

train_ds = dataset_from_dict({'col1': [1, 2, 3, 4, 10, 11]}, 'col1')
test_ds = dataset_from_dict({'col1': [4, 3, 5, 6, 7]}, 'col1')
check_obj = IndexTrainTestLeakage()
check_obj.run(train_ds, test_ds)

#%%

train_ds = dataset_from_dict({'col1': [1, 2, 3, 4, 10, 11]}, 'col1')
test_ds = dataset_from_dict({'col1': [4, 3, 5, 6, 7]}, 'col1')
check_obj = IndexTrainTestLeakage(n_to_show=1)
check_obj.run(train_ds, test_ds)

#%%
# Synthetic example without index leakage
# =======================================

train_ds = dataset_from_dict({'col1': [1, 2, 3, 4, 10, 11]}, 'col1')
test_ds = dataset_from_dict({'col1': [20, 21, 5, 6, 7]}, 'col1')
check_obj = IndexTrainTestLeakage()
check_obj.run(train_ds, test_ds)
