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
"""Utils module for calculating embeddings for text."""

import pandas as pd


def get_default_embeddings(text: pd.Series, model: str = 'miniLM', file_path: str = 'embeddings.csv') -> pd.DataFrame:
    """
    Get default embeddings for the dataset.

    Parameters
    ----------
    text : pd.Series
        The text to get embeddings for.
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
        embeddings = model.encode(text)
    elif model == 'open_ai':
        try:
            import openai
        except ImportError as e:
            raise ImportError(
                'get_default_embeddings with model="open_ai" requires the openai python package. '
                'To get it, run "pip install openai".'
            ) from e

        from tenacity import retry, stop_after_attempt, wait_random_exponential

        @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
        def _get_embedding_with_backoff(list_of_strings):
            return openai.Embedding.create(input=list_of_strings, model="text-embedding-ada-002")['data']

        batch_size = 500
        embeddings = []
        clean_text = [clean_special_chars(x) for x in text]
        for idx, sub_list in enumerate([clean_text[x:x+batch_size] for x in range(0, len(text), batch_size)]):
            open_ai_response = _get_embedding_with_backoff(sub_list)
            for x in open_ai_response:
                embeddings.append(x['embedding'])
            print(f'Finished batch {idx+1} out of {len(text)//batch_size + 1}')
    else:
        raise ValueError(f'Unknown model type: {model}')
    embeddings = pd.DataFrame(embeddings, index=text.index)
    if file_path:
        embeddings.to_csv(file_path, index=True)
    return embeddings


def clean_special_chars(text):
    special_chars = '!@#$%^&*()_+{}|:"<>?~`-=[]\;\',./'
    for char in special_chars:
        text = text.replace(char, '')
    text = text.replace("\n", " ")
    text = text.replace("<br />", " ")
    return text