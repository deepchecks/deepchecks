# ----------------------------------------------------------------------------
# Copyright (C) 2021-2023 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""Utils module for calculating embeddings or completion for text."""
from typing import List, Optional, Sequence

import pandas as pd
from tqdm import tqdm


def calculate_embeddings_for_text(text: pd.Series, model: str = 'miniLM',
                                  file_path: Optional[str] = 'embeddings.csv') -> pd.DataFrame:
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
    file_path : Optional[str], default 'embeddings.csv'
        If given, the embeddings will be saved to the given file path.

    Returns
    -------
        pd.DataFrame
            The embeddings for the dataset.
    """
    if model == 'miniLM':
        try:
            import sentence_transformers  # pylint: disable=import-outside-toplevel
        except ImportError as e:
            raise ImportError(
                'get_default_embeddings with model="miniLM" requires the sentence_transformers python package. '
                'To get it, run "pip install sentence_transformers".') from e

        model = sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(text)
    elif model == 'open_ai':
        try:
            import openai  # pylint: disable=import-outside-toplevel
        except ImportError as e:
            raise ImportError('get_default_embeddings with model="open_ai" requires the openai python package. '
                              'To get it, run "pip install openai".') from e

        from tenacity import (retry, stop_after_attempt,  # pylint: disable=import-outside-toplevel
                              wait_random_exponential)

        @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
        def _get_embedding_with_backoff(list_of_strings):
            return openai.Embedding.create(input=list_of_strings, model='text-embedding-ada-002')['data']

        batch_size = 500
        embeddings = []
        clean_text = [_clean_special_chars(x) for x in text]
        for sub_list in tqdm([clean_text[x:x + batch_size] for x in range(0, len(text), batch_size)],
                             desc='Calculating Embeddings '):
            open_ai_response = _get_embedding_with_backoff(sub_list)
            for x in open_ai_response:
                embeddings.append(x['embedding'])
    else:
        raise ValueError(f'Unknown model type: {model}')
    embeddings = pd.DataFrame(embeddings, index=text.index)
    if file_path is not None:
        embeddings.to_csv(file_path, index=True)
    return embeddings


def call_open_ai_completion_api(inputs: Sequence[str], max_tokens=200, batch_size=20,  # api limit of 20 requests
                                model: str = 'text-davinci-003', temperature: float = 0.5) -> List[str]:
    """
    Call the open ai completion api with the given inputs batch by batch.

    Parameters
    ----------
    inputs : Sequence[str]
        The inputs to send to the api.
    max_tokens : int, default 200
        The maximum number of tokens to return for each input.
    batch_size : int, default 20
        The number of inputs to send in each batch.
    model : str, default 'text-davinci-003'
        The model to use for the question answering task. For more information about the models, see:
        https://beta.openai.com/docs/api-reference/models
    temperature : float, default 0.5
        The temperature to use for the question answering task. For more information about the temperature, see:
        https://beta.openai.com/docs/api-reference/completions/create-completion

    Returns
    -------
    List[str]
        The answers for the questions.
    """
    try:
        import openai  # pylint: disable=import-outside-toplevel
    except ImportError as e:
        raise ImportError('question_answering_open_ai requires the openai python package. '
                          'To get it, run "pip install openai".') from e

    from tenacity import retry, stop_after_attempt, wait_random_exponential  # pylint: disable=import-outside-toplevel

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(15))
    def _get_answers_with_backoff(questions_in_context):
        return openai.Completion.create(engine=model, prompt=questions_in_context,
                                        max_tokens=max_tokens, temperature=temperature)

    answers = []
    for sub_list in tqdm([inputs[x:x + batch_size] for x in range(0, len(inputs), batch_size)],
                         desc=f'Calculating Responses (Total of {len(inputs)})'):
        open_ai_responses = _get_answers_with_backoff(sub_list)
        choices = sorted(open_ai_responses['choices'], key=lambda x: x['index'])
        answers = answers + [choice['text'] for choice in choices]
    return answers


def _clean_special_chars(text):
    special_chars = r'!@#$%^&*()_+{}|:"<>?~`-=[]\;\',./'
    for char in special_chars:
        text = text.replace(char, '')
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    text = text.replace('<br />', ' ')
    return text
