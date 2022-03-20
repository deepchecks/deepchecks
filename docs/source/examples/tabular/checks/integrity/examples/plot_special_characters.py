# -*- coding: utf-8 -*-
"""
Special Characters
******************
"""

#%%

from deepchecks.tabular.checks import SpecialCharacters
import pandas as pd

#%%

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
SpecialCharacters().run(dataframe)
