import openai
import pandas as pd
import numpy as np
from datasets import load_dataset
from deepchecks.nlp.text_data import TextData
from deepchecks.nlp.suites import full_suite
from deepchecks.nlp.checks import *
from deepchecks.nlp.utils.embeddings import clean_special_chars

dataset = load_dataset('tweet_eval', 'emoji', split='train')
train_data = pd.DataFrame(dataset)
dataset = load_dataset('tweet_eval', 'emoji', split='test')
test_data = pd.DataFrame(dataset)
full_data = pd.concat([train_data,test_data])
full_data.reset_index(drop = True, inplace=True)

full_data.head()

from deepchecks.nlp.utils.embeddings import get_default_embeddings
import openai
import os

file_path = '/Users/nadav/Desktop/Experiments/DeepChecks_Package/NLP/tweet_eval_open_ai_embedding.csv'
if os.path.exists(file_path):
    embedding = pd.read_csv(file_path, index_col=0).to_numpy()
else:
    openai.api_key = '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'
    ds_full = TextData(raw_text=full_data['text'].values.tolist(), label=full_data['label'].values.tolist(), task_type='text_classification')
    embedding = get_default_embeddings(ds_full, model='open_ai', file_path=file_path).to_numpy()

print(embedding.shape)
print(embedding[0,:])

BASE_SIZE = 1000


def train_test_data(only_train_prop, only_test_prop, not_in_both_prop, drifted_ratio=4):
    train_drifted = full_data[full_data['label'].isin(only_train_prop)].sample(BASE_SIZE) if len(
        only_train_prop) > 0 else full_data.sample(2)
    test_drifted = full_data[full_data['label'].isin(only_test_prop)].sample(BASE_SIZE) if len(
        only_test_prop) > 0 else full_data.sample(2)
    drifted_indexes_train, drifted_indexes_test = list(train_drifted.index), list(test_drifted.index)

    remaining_data = full_data[~full_data['label'].isin(not_in_both_prop + only_test_prop + only_train_prop)].sample(
        BASE_SIZE * drifted_ratio * 2)
    print(train_drifted.shape, test_drifted.shape, remaining_data.shape)

    train_remaining = remaining_data.sample(frac=0.7)
    train_drifted = pd.concat([train_drifted, train_remaining])
    test_drifted = pd.concat([test_drifted, remaining_data.drop(train_remaining.index)])

    return train_drifted, test_drifted, drifted_indexes_train, drifted_indexes_test


# create train test split
only_train_prop = []  # funny + images
only_test_prop = [2, 10]
not_in_both_prop = [11, 12, 17, 18, ]

train_drifted, test_drifted, drifted_indexes_train, drifted_indexes_test = train_test_data(only_train_prop,
                                                                                           only_test_prop,
                                                                                           not_in_both_prop,
                                                                                           drifted_ratio=1)

# embedding split
train_embedding = embedding[train_drifted.index, :]
test_embedding = embedding[test_drifted.index, :]


ds_train = TextData(raw_text=train_drifted['text'].values.tolist(), label=train_drifted['label'].values.tolist(),
                    task_type='text_classification', index=train_drifted.index)
ds_test = TextData(raw_text=test_drifted['text'].values.tolist(), label=test_drifted['label'].values.tolist(),
                   task_type='text_classification', index=test_drifted.index)

check = TextEmbeddingsDrift(min_meaningful_drift_score=0, min_feature_importance=0.02,
                            train_indexes_to_highlight=drifted_indexes_train, test_indexes_to_highlight=drifted_indexes_test)
result = check.run(train_dataset=ds_train, test_dataset=ds_test, train_embeddings=train_embedding, test_embeddings=test_embedding)