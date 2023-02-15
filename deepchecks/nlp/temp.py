import datetime
import os
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import tensorflow as tf

base = '/Users/nadav/Desktop/Experiments/DeepChecks_Package/NLP/UseCases/'
data = pd.read_csv(base + 'bbc-news-data.csv', sep='\t')

categories = data['category'].value_counts().keys()
num_for_each_in_train = [200, 250, 300, 200, 200]  # train has more politics and less sport
index_train = []
for idx, val in enumerate(num_for_each_in_train):
    index_train += list(data[data['category'] == categories[idx]].sample(val).index)

train, test = data[data.index.isin(index_train)].sample(100), data[~data.index.isin(index_train)].sample(100)

from deepchecks.nlp import TextData

test_ds = TextData(raw_text=test.title, label=test.category, task_type='text_classification', index= test.index, additional_data=test[['category', 'filename']])
# train_ds = TextData(raw_text=train.title, label=train.category, task_type='text_classification', index= train.index, meta_data=train[['category', 'filename']])
test_ds.head()
from deepchecks.nlp.checks import WeakSegmentsPerformance

check = WeakSegmentsPerformance()
check.run(test_ds, predictions=test_ds.label)