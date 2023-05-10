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
from typing import List, Sequence

from tqdm import tqdm


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
