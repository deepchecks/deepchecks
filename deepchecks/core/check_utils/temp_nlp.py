import pandas as pd
import numpy as np
from datasets import load_dataset
from deepchecks.nlp.text_data import TextData

dataset = load_dataset('trec', split='train')
train_data = pd.DataFrame(dataset)
dataset = load_dataset('trec', split='test')
test_data = pd.DataFrame(dataset)
full_data = pd.concat([train_data,test_data])
full_data.reset_index(drop = True, inplace=True)
full_data.head()


from deepchecks.nlp.utils.embeddings import get_default_embeddings
import os

file_path = '/Users/nadav/Desktop/Experiments/DeepChecks_Package/NLP/open_ai_embedding.csv'
if os.path.exists(file_path):
    embedding = pd.read_csv(file_path, index_col=0).to_numpy()
else:
    # openai.api_key = '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'
    ds_full = TextData(raw_text=full_data['text'].values.tolist(), label=full_data['coarse_label'].values.tolist(), task_type='text_classification')
    embedding = get_default_embeddings(ds_full, model='open_ai', file_path=file_path).to_numpy()


only_train_prop = [32,33,34,35]
only_test_prop = [28,29,30]
not_in_both = [36,31]

train_drifted = full_data[full_data['fine_label'].isin(only_train_prop)]
test_drifted = full_data[full_data['fine_label'].isin(only_test_prop)]
drifted_indexes_train, drifted_indexes_test = list(train_drifted.index), list(test_drifted.index)

remaining_data = full_data[~full_data['fine_label'].isin(not_in_both)]
remaining_data = remaining_data.drop(train_drifted.index)
remaining_data = remaining_data.drop(test_drifted.index)
print(remaining_data.shape)

train_remaining = remaining_data.sample(frac = 0.6)
train_drifted = pd.concat([train_drifted, train_remaining])
test_drifted = pd.concat([test_drifted, remaining_data.drop(train_remaining.index)])

train_embedding = embedding[train_drifted.index, :]
test_embedding = embedding[test_drifted.index, :]

from deepchecks.nlp.checks import *
ds_train = TextData(raw_text=train_drifted['text'].values.tolist(), label=train_drifted['fine_label'].values.tolist(),
                    task_type='text_classification', index=train_drifted.index)
ds_test = TextData(raw_text=test_drifted['text'].values.tolist(), label=test_drifted['fine_label'].values.tolist(),
                   task_type='text_classification', index=test_drifted.index)

check = TextEmbeddingsDrift(min_meaningful_drift_score=0, min_feature_importance=0.02,
                            train_indexes_to_highlight=drifted_indexes_train, test_indexes_to_highlight=drifted_indexes_test)
result = check.run(train_dataset=ds_train, test_dataset=ds_test, train_embeddings=train_embedding, test_embeddings=test_embedding)
result