# -*- coding: utf-8 -*-
"""
.. _plot_tabular_special_chars:

Special Characters
******************

This notebook provides an overview for using and understanding the Special Characters check.

**Structure:**

* `What is the Special Characters check <#what-is-special-characters>`__
* `Generate data <#generate-data>`__
* `Run the check <#run-the-check>`__
"""

# %%
# What is the Special Characters check
# ======================================
# The ``SpecialCharacters`` check search in column[s] for values that contains only special characters.


#%%

import pandas as pd

from deepchecks.tabular.checks import SpecialCharacters

#%%
# Generate Data
# ==============
data = {'col1': [' ', '!', '"', '#', '$', '%', '&', '\'','(', ')',
                 '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', 
                 '>', '?', '@', '[', ']', '\\', '^', '_', '`', '{',
                 '}', '|', '~', '\n'],
        'col2':['v', 'v', 'v', 'v4', 'v5', 'v6', 'v7', 'v8','v9','v10', 
                 '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', 
                 '>', '?', '@', '[', ']', '\\', '^', '_', '`', '{',
                 '}', '|', '~', '\n'],
        'col3': [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,11,1,'???#',1,1,1,1,1,1,1,1,1,1,1],
        'col4': [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,11,1,1,1,1,1,1,1,1,1,1,1,1,1],
        'col5': ['valid1','valid2','valid3','valid4','valid5','valid6','valid7',
                 'valid8','valid9','valid10','valid11','valid12',
                'valid13','valid14','inval!d15','valid16','valid17','valid18',
                 'valid19','valid20','valid21','valid22','valid23','valid24','valid25',
                'valid26', 'valid27','valid28','valid29','valid30','valid31','32','33','34']}

dataframe = pd.DataFrame(data=data)

#%%
# Run the check
# ===============
SpecialCharacters().run(dataframe)
