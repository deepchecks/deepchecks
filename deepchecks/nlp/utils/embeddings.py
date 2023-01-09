# ----------------------------------------------------------------------------
# Copyright (C) 2021-2022 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""Module containing NLP embedding utils."""
# TODO: Prototype, go over and make sure code+docs+tests are good

__all__ = ['get_default_embeddings']

import warnings
import asyncio

import numpy as np
import pandas as pd

from deepchecks.nlp.text_data import TextData


def get_default_embeddings(dataset: TextData, model: str = 'miniLM', file_path: str = 'embeddings.csv') -> pd.DataFrame:
    """
    Get default embeddings for the dataset.

    Parameters
    ----------
    dataset
    model : str, default 'miniLM'
        The type of embeddings to return. Can be either 'miniLM' or 'open_ai'.
        For 'open_ai' option, the model used is 'text-embedding-ada-002' and requires to first set an open ai api key
        by using the command openai.api_key = YOUR_API_KEY
    file_path : str, default 'embeddings.csv'
        If given, the embeddings will be saved to the given file path.

    Returns
    -------
        pd.DataFrame
            The embeddings for the dataset.
    """
    if model == 'miniLM':
        try:
            import sentence_transformers
        except ImportError as e:
            raise ImportError(
                'get_default_embeddings with model="miniLM" requires the sentence_transformers python package. '
                'To get it, run "pip install sentence_transformers".'
            ) from e

        model = sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(dataset.text)
        embeddings = pd.DataFrame(embeddings, index=dataset.index)
    elif model == 'open_ai':
        try:
            import openai
        except ImportError as e:
            raise ImportError(
                'get_default_embeddings with model="open_ai" requires the openai python package. '
                'To get it, run "pip install openai".'
            ) from e

        async def _get_open_ai_embedding(text, model="text-embedding-ada-002"):
            text = text.replace("\n", " ")
            text = text.replace("<br />", " ")
            text = clean_special_chars(text)
            try:
                embedding = openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']
            except:
                embedding = np.zeros(1536)
                warnings.warn(f"Failed to get embedding for sample")
            return embedding

        from datetime import datetime

        start = datetime.now()

        embeddings = [[]] * len(dataset.text)
        for i in range(dataset.n_samples): # Doesn't actually works
            embeddings[i] = asyncio.run(_get_open_ai_embedding(dataset.text[i]))

        embeddings = pd.DataFrame(embeddings, index=dataset.index)

        print(f"Time to get embeddings: {datetime.now() - start}")
    else:
        raise ValueError(f"Invalid type {type}. Can be either 'miniLM' or 'open_ai'")

    if file_path:
        embeddings.to_csv(file_path, index=dataset.index)

    return embeddings




def clean_special_chars(text):
    special_chars = '!@#$%^&*()_+{}|:"<>?~`-=[]\;\',./'
    for char in special_chars:
        text = text.replace(char, '')
    return text