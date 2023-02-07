import pandas as pd
import numpy as np
from datasets import load_dataset
from deepchecks.nlp.text_data import TextData
from deepchecks.nlp.suites import full_suite
from deepchecks.nlp.checks import *
from deepchecks.core.check_utils.multivariate_drift_utils import create_outlier_display, create_embedding_display


dataset = load_dataset('trec', split='train')
train_data = pd.DataFrame(dataset)
dataset = load_dataset('trec', split='test')
test_data = pd.DataFrame(dataset)
full_data = pd.concat([train_data,test_data])
full_data.reset_index(drop = True, inplace=True)

from deepchecks.nlp.utils.embeddings import get_default_embeddings
import openai
import os

file_path = '/Users/nadav/Desktop/Experiments/DeepChecks_Package/NLP/trec_open_ai_embedding.csv'
if os.path.exists(file_path):
    embedding = pd.read_csv(file_path, index_col=0).to_numpy()
else:
    openai.api_key = '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'
    ds_full = TextData(raw_text=full_data['text'].values.tolist(), label=full_data['coarse_label'].values.tolist(), task_type='text_classification')
    embedding = get_default_embeddings(ds_full, model='open_ai', file_path=file_path).to_numpy()

print(embedding.shape)
print(embedding[0,:])

BASE_SIZE = 1000


def create_drifted_data(prop_in_full, outlier_prop, outlier_ratio=0.05):
    data = full_data[full_data['fine_label'].isin(prop_in_full)]
    outlier_data = full_data[full_data['fine_label'].isin(outlier_prop)].sample(int(outlier_ratio * len(data)))

    print(data.shape, outlier_data.shape)

    data = pd.concat([data, outlier_data])
    return data, outlier_data.index


# create train test split
data_props = list(range(37, 50, 1))  # 'NUM' (5): Numeric value.
outlier_props = [28, 5]


data_for_check, outlier_indexes = create_drifted_data(data_props, outlier_props)

# embedding split
data_embedding = embedding[data_for_check.index, :]

path = '/Users/nadav/Desktop/Experiments/DeepChecks_Package/NLP/logs/trec-example/outliers'
indexes_to_highlight = {'real outlier samples': outlier_indexes}
create_embedding_display(data_for_check['text'],data_for_check['text'], data_embedding, data_embedding, path=path, indexes_to_highlight=indexes_to_highlight)