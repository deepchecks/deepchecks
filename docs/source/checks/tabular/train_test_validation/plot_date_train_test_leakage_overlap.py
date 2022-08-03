# -*- coding: utf-8 -*-
"""
.. _plot_tabular_date_train_test_validation_leakage_overlap:

Date Train Test Leakage Overlap
******************************************
"""

#%%

from datetime import datetime

import pandas as pd

from deepchecks.tabular import Dataset, Suite
from deepchecks.tabular.checks import DateTrainTestLeakageOverlap


def dataset_from_dict(d: dict, datetime_name: str = None) -> Dataset:
    dataframe = pd.DataFrame(data=d)
    return Dataset(dataframe, datetime_name=datetime_name)

#%%
# Synthetic example dates before last training
# ============================================

train_ds = dataset_from_dict({'col1': [
        datetime(2021, 10, 1, 0, 0),
        datetime(2021, 10, 1, 0, 0),
        datetime(2021, 10, 1, 0, 0),
        datetime(2021, 10, 2, 0, 0),
        datetime(2021, 10, 2, 0, 0),
        datetime(2021, 10, 2, 0, 0),
        datetime(2021, 10, 3, 0, 0),
        datetime(2021, 10, 3, 0, 0),
        datetime(2021, 10, 3, 0, 0),
        datetime(2021, 10, 4, 0, 0),
        datetime(2021, 10, 4, 0, 0),
        datetime(2021, 10, 4, 0, 0),
        datetime(2021, 10, 5, 0, 0),
        datetime(2021, 10, 5, 0, 0)
    ]}, 'col1')
test_ds = dataset_from_dict({'col1': [
        datetime(2021, 9, 4, 0, 0),
        datetime(2021, 10, 6, 0, 0),
        datetime(2021, 10, 6, 0, 0),
        datetime(2021, 10, 7, 0, 0),
        datetime(2021, 10, 7, 0, 0),
        datetime(2021, 10, 8, 0, 0),
        datetime(2021, 10, 8, 0, 0),
        datetime(2021, 10, 9, 0, 0),
        datetime(2021, 10, 9, 0, 0)
    ]}, 'col1')

DateTrainTestLeakageOverlap().run(train_dataset=train_ds, test_dataset=test_ds)

#%%
# Synthetic example no date leakage
# =================================

train_ds = dataset_from_dict({'col1': [
        datetime(2021, 10, 3, 0, 0),
        datetime(2021, 10, 3, 0, 0),
        datetime(2021, 10, 4, 0, 0),
        datetime(2021, 10, 4, 0, 0),
        datetime(2021, 10, 4, 0, 0),
        datetime(2021, 10, 5, 0, 0),
        datetime(2021, 10, 5, 0, 0)
    ]}, 'col1')
test_ds = dataset_from_dict({'col1': [
        datetime(2021, 11, 4, 0, 0),
        datetime(2021, 11, 4, 0, 0),
        datetime(2021, 11, 5, 0, 0),
        datetime(2021, 11, 6, 0, 0),

    ]}, 'col1')

DateTrainTestLeakageOverlap().run(train_dataset=train_ds, test_dataset=test_ds)
