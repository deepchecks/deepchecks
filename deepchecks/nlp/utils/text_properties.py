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
"""Module containing the text properties for the NLP module."""
import pathlib
import pickle as pkl
import re
import string
import warnings
from collections import defaultdict
from importlib.util import find_spec
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import textblob
import torch.cuda
from nltk import corpus
from nltk import download as nltk_download
from nltk import sent_tokenize, word_tokenize
from tqdm import tqdm
from typing_extensions import TypedDict

from deepchecks.core.errors import DeepchecksValueError
from deepchecks.nlp.utils.text import cut_string, hash_text, normalize_text, remove_punctuation
from deepchecks.nlp.utils.text_properties_models import get_cmudict_dict, get_fasttext_model, get_transformer_pipeline
from deepchecks.utils.function import run_available_kwargs
from deepchecks.utils.strings import SPECIAL_CHARACTERS, format_list

__all__ = ['calculate_builtin_properties', 'get_builtin_properties_types']

from deepchecks.utils.validation import is_sequence_not_str

DEFAULT_SENTENCE_SAMPLE_SIZE = 300
MAX_CHARS = 512  # Bert accepts max of 512 tokens, so without counting tokens we go for the lower bound.
# all SPECIAL_CHARACTERS - all string.punctuation except for <>@[]^_`{|}~ - all whitespace
NON_PUNCTUATION_SPECIAL_CHARS = frozenset(set(SPECIAL_CHARACTERS) - set(r"""!"#$%&'()*+,-./:;=?\@""")
                                          - set(string.whitespace))

textblob_cache = {}
words_cache = {}
sentences_cache = {}
secret_cache = {}


def _split_to_words_with_cache(text: str) -> List[str]:
    """Tokenize a text into words and cache the result."""
    hash_key = hash_text(text)
    if hash_key not in words_cache:
        words = re.split(r'\W+', normalize_text(text, remove_stops=False, ignore_whitespace=False))
        words = [w for w in words if w]  # remove empty strings
        words_cache[hash_key] = words
    return words_cache[hash_key]


def _split_to_sentences_with_cache(text: str) -> Union[List[str], None]:
    """Tokenize a text into sentences and cache the result."""
    hash_key = hash_text(text)
    if hash_key not in sentences_cache:
        if not nltk_download('punkt', quiet=True):
            _warn_if_missing_nltk_dependencies('punkt', 'property')
            return None
        sentences_cache[hash_key] = sent_tokenize(text)
    return sentences_cache[hash_key]


def _sample_for_property(text: str, mode: str = 'words', limit: int = 10000, return_as_list=False,
                         random_seed: int = 42) -> Union[str, List[str]]:
    """Get a sample a single text sample for a text property.

    Parameters
    ----------
    text : str
        The text to sample from.
    mode : str, default 'words'
        The mode to sample in. Can be either 'words' or 'sentences'.
    limit : int, default 10000
        The maximum number of words or sentences to sample.
    """
    np.random.seed(random_seed)
    if pd.isna(text):
        return None

    if mode == 'words':
        all_units = _split_to_words_with_cache(text)
        if len(all_units) > limit:
            all_units = np.random.choice(all_units, size=limit, replace=False)
    elif mode == 'sentences':
        all_units = _split_to_sentences_with_cache(text)
        if len(all_units) > limit:
            all_units = np.random.choice(all_units, size=limit, replace=False)
    else:
        raise DeepchecksValueError(f'Unexpected mode - {mode}')

    return ' '.join(all_units) if not return_as_list else list(all_units)


def _warn_if_missing_nltk_dependencies(dependency: str, property_name: str):
    """Warn if NLTK dependency is missing."""
    warnings.warn(f'NLTK {dependency} not found, {property_name} cannot be calculated.'
                  ' Please check your internet connection.', UserWarning)


def text_length(text: str) -> int:
    """Return text length."""
    return len(text)


def average_word_length(text: str) -> float:
    """Return average word length."""
    words = _split_to_words_with_cache(text)
    return np.mean([len(word) for word in words]) if words else 0


def percentage_special_characters(text: str) -> float:
    """Return percentage of special characters (as float between 0 and 1)."""
    return len([c for c in text if c in NON_PUNCTUATION_SPECIAL_CHARS]) / len(text) if len(text) != 0 else 0


def percentage_punctuation(text: str) -> float:
    """Return percentage of punctuation (as float between 0 and 1)."""
    return len([c for c in text if c in string.punctuation]) / len(text) if len(text) != 0 else 0


def max_word_length(text: str) -> int:
    """Return max word length."""
    words = _split_to_words_with_cache(text)
    return max(len(w) for w in words) if words else 0


def language(
        text: str,
        lang_certainty_threshold: float = 0.8,
        fasttext_model: Optional[Dict[object, Any]] = None
) -> Union[str, None]:
    """Return text language, represented as a string."""
    if not text:
        return None
    # Load the model if it wasn't received as a parameter. This is done to avoid loading the model
    # each time the function is called.
    if fasttext_model is None:
        fasttext_model = get_fasttext_model()

    # Predictions are the first prediction (k=1), only if the probability is above the threshold
    prediction = fasttext_model.predict(text.replace('\n', ' '), k=1, threshold=lang_certainty_threshold)[0]
    # label is empty for detection below threshold:
    language_code = prediction[0].replace('__label__', '') if prediction else None

    if language_code == 'eng':  # both are english but different labels
        return 'en'
    return language_code


def english_text(
        text: str,
        lang_certainty_threshold: float = 0.8,
        fasttext_model: Optional[Dict[object, Any]] = None,
        language_property_result: Optional[str] = None
) -> Union[bool, None]:
    """Return whether text is in English or not."""
    if not text:
        return None
    if language_property_result is None:
        language_property_result = language(text, lang_certainty_threshold, fasttext_model)
    return language_property_result == 'en'


def sentiment(text: str) -> float:
    """Return float representing sentiment."""
    hash_key = hash_text(text)
    if textblob_cache.get(hash_key) is None:
        # TextBlob uses only the words and not the relations between them, so we can sample the text
        # to speed up the process:
        words = _sample_for_property(text, mode='words')
        textblob_cache[hash_key] = textblob.TextBlob(words).sentiment
    return textblob_cache.get(hash_key).polarity


def subjectivity(text: str) -> float:
    """Return float representing subjectivity."""
    hash_key = hash_text(text)
    if textblob_cache.get(hash_key) is None:
        # TextBlob uses only the words and not the relations between them, so we can sample the text
        # to speed up the process:
        words = _sample_for_property(text, mode='words')
        textblob_cache[hash_key] = textblob.TextBlob(words).sentiment
    return textblob_cache.get(hash_key).subjectivity


def predict_on_batch(text_batch: Sequence[str], classifier,
                     output_formatter: Callable[[Dict[str, Any]], float]) -> Sequence[float]:
    """Return prediction of huggingface Pipeline classifier."""
    # TODO: make this way smarter, and not just a hack. Count tokens, for a start. Then not just sample sentences.
    # If text is longer than classifier context window, sample it:
    text_list_to_predict = []
    reduced_batch_size = len(text_batch)  # Initialize the reduced batch size
    retry_count = 0

    for text in text_batch:
        if len(text) > MAX_CHARS:
            sentences = _sample_for_property(text, mode='sentences', limit=10, return_as_list=True)
            text_to_use = ''
            for sentence in sentences:
                if len(text_to_use) + len(sentence) > MAX_CHARS:
                    break
                text_to_use += sentence + '. '

            # if even one sentence is too long, use part of the first one:
            if len(text_to_use) == 0:
                text_to_use = cut_string(sentences[0], MAX_CHARS)
            text_list_to_predict.append(text_to_use)
        else:
            text_list_to_predict.append(text)

    while reduced_batch_size >= 1:
        try:
            if reduced_batch_size == 1 or retry_count == 3:
                results = []
                for text in text_list_to_predict:
                    try:
                        v = classifier(text)[0]
                        results.append(output_formatter(v))
                    except Exception:  # pylint: disable=broad-except
                        results.append(np.nan)
                return results  # Return the results if prediction is successful

            v_list = classifier(text_list_to_predict, batch_size=reduced_batch_size)
            results = []

            for v in v_list:
                results.append(output_formatter(v))

            return results  # Return the results if prediction is successful

        except Exception:  # pylint: disable=broad-except
            reduced_batch_size = max(reduced_batch_size // 2, 1)  # Reduce the batch size by half
            retry_count += 1

    return [np.nan] * len(text_batch)  # Prediction failed, return NaN values for the original batch size


TOXICITY_CALIBRATOR = pathlib.Path(__file__).absolute().parent / 'assets' / 'toxicity_calibrator.pkl'
TOXICITY_MODEL_NAME = 'SkolkovoInstitute/roberta_toxicity_classifier'
TOXICITY_MODEL_NAME_ONNX = 'Deepchecks/roberta_toxicity_classifier_onnx'
FLUENCY_MODEL_NAME = 'prithivida/parrot_fluency_model'
FLUENCY_MODEL_NAME_ONNX = 'Deepchecks/parrot_fluency_model_onnx'
FORMALITY_MODEL_NAME = 's-nlp/roberta-base-formality-ranker'
FORMALITY_MODEL_NAME_ONNX = 'Deepchecks/roberta_base_formality_ranker_onnx'


def toxicity(
        text_batch: Sequence[str],
        device: Optional[str] = None,
        models_storage: Union[pathlib.Path, str, None] = None,
        use_onnx_models: bool = True,
        toxicity_classifier: Optional[object] = None
) -> Sequence[float]:
    """Return float representing toxicity."""
    if toxicity_classifier is None:
        use_onnx_models = _validate_onnx_model_availability(use_onnx_models, device)
        model_name = TOXICITY_MODEL_NAME_ONNX if use_onnx_models else TOXICITY_MODEL_NAME
        toxicity_classifier = get_transformer_pipeline(
            property_name='toxicity', model_name=model_name, device=device,
            models_storage=models_storage, use_onnx_model=use_onnx_models)

    class UnitModel:
        """A model that does nothing."""

        @staticmethod
        def predict(x):
            return x

    try:
        with open(TOXICITY_CALIBRATOR, 'rb') as f:
            toxicity_calibrator = pkl.load(f)
    except Exception:  # pylint: disable=broad-except
        toxicity_calibrator = UnitModel()

    def output_formatter(v):
        score = v['score'] if (v['label'] == 'toxic') else 1 - v['score']
        return toxicity_calibrator.predict([score])[0]

    return predict_on_batch(text_batch, toxicity_classifier, output_formatter)


def fluency(
        text_batch: Sequence[str],
        device: Optional[str] = None,
        models_storage: Union[pathlib.Path, str, None] = None,
        use_onnx_models: bool = True,
        fluency_classifier: Optional[object] = None
) -> Sequence[float]:
    """Return float representing fluency."""
    if fluency_classifier is None:
        use_onnx_models = _validate_onnx_model_availability(use_onnx_models, device)
        model_name = FLUENCY_MODEL_NAME_ONNX if use_onnx_models else FLUENCY_MODEL_NAME
        fluency_classifier = get_transformer_pipeline(
            property_name='fluency', model_name=model_name, device=device,
            models_storage=models_storage, use_onnx_model=use_onnx_models)

    def output_formatter(v):
        return v['score'] if v['label'] == 'LABEL_1' else 1 - v['score']

    return predict_on_batch(text_batch, fluency_classifier, output_formatter)


def formality(
        text_batch: Sequence[str],
        device: Optional[str] = None,
        models_storage: Union[pathlib.Path, str, None] = None,
        use_onnx_models: bool = True,
        formality_classifier: Optional[object] = None
) -> Sequence[float]:
    """Return float representing formality."""
    if formality_classifier is None:
        use_onnx_models = _validate_onnx_model_availability(use_onnx_models, device)
        model_name = FORMALITY_MODEL_NAME_ONNX if use_onnx_models else FORMALITY_MODEL_NAME
        formality_classifier = get_transformer_pipeline(
            property_name='formality', model_name=model_name, device=device,
            models_storage=models_storage, use_onnx_model=use_onnx_models)

    def output_formatter(v):
        return v['score'] if v['label'] == 'formal' else 1 - v['score']

    return predict_on_batch(text_batch, formality_classifier, output_formatter)


def lexical_density(text: str) -> float:
    """Return a float representing lexical density.

    Lexical density is the percentage of unique words in a given text. For more
    information: https://en.wikipedia.org/wiki/Lexical_density
    """
    if pd.isna(text):
        return np.nan
    if not nltk_download('punkt', quiet=True):
        _warn_if_missing_nltk_dependencies('punkt', 'Lexical Density')
        return np.nan

    all_words = _split_to_words_with_cache(text)
    if len(all_words) == 0:
        return np.nan
    total_unique_words = len(set(all_words))
    return round(total_unique_words / len(all_words), 2)


def unique_noun_count(text: Sequence[str]) -> int:
    """Return the number of unique noun words in the text."""
    if pd.isna(text):
        return np.nan
    if not nltk_download('averaged_perceptron_tagger', quiet=True):
        _warn_if_missing_nltk_dependencies('averaged_perceptron_tagger', 'Unique Noun Count')
        return np.nan

    unique_words_with_tags = set(textblob.TextBlob(text).tags)
    return sum(1 for (_, tag) in unique_words_with_tags if tag.startswith('N'))


def readability_score(text: str, cmudict_dict: dict = None) -> float:
    """Return a float representing the Flesch Reading-Ease score per text sample.

    In the Flesch reading-ease test, higher scores indicate material that is easier to read
    whereas lower numbers mark texts that are more difficult to read. For more information:
    https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests#Flesch_reading_ease
    """
    if pd.isna(text):
        return np.nan
    if cmudict_dict is None:
        if not nltk_download('cmudict', quiet=True):
            _warn_if_missing_nltk_dependencies('cmudict', 'Reading Ease')
            return np.nan
        cmudict_dict = corpus.cmudict.dict()
    text_sentences = _sample_for_property(text, mode='sentences', limit=DEFAULT_SENTENCE_SAMPLE_SIZE,
                                          return_as_list=True)
    sentence_count = len(text_sentences)
    words = _split_to_words_with_cache(text)
    word_count = len(words)
    syllable_count = sum([len(cmudict_dict[word]) for word in words if word in cmudict_dict])

    if word_count != 0 and sentence_count != 0 and syllable_count != 0:
        avg_syllables_per_word = syllable_count / word_count
        avg_words_per_sentence = word_count / sentence_count
        flesch_reading_ease = 206.835 - (1.015 * avg_words_per_sentence) - (84.6 * avg_syllables_per_word)
        return round(flesch_reading_ease, 3)
    else:
        return np.nan


def average_words_per_sentence(text: str) -> float:
    """Return the average words per sentence in the text."""
    if pd.isna(text):
        return np.nan
    text_sentences = _sample_for_property(text, mode='sentences', limit=DEFAULT_SENTENCE_SAMPLE_SIZE,
                                          return_as_list=True)
    if text_sentences:
        text_sentences = [remove_punctuation(sent) for sent in text_sentences]
        total_words = sum([len(_split_to_words_with_cache(sentence)) for sentence in text_sentences])
        return round(total_words / len(text_sentences), 3)
    else:
        return np.nan


def unique_urls_count(text: str) -> int:
    """Return the number of unique URLS in the text."""
    if pd.isna(text):
        return np.nan
    url_pattern = r'https?:\/\/(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
    return len(set(re.findall(url_pattern, text)))


def urls_count(text: str) -> int:
    """Return the number of URLS in the text."""
    if pd.isna(text):
        return np.nan
    url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
    return len(re.findall(url_pattern, text))


def unique_email_addresses_count(text: str) -> int:
    """Return the number of unique email addresses in the text."""
    if pd.isna(text):
        return np.nan
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
    return len(set(re.findall(email_pattern, text)))


def email_addresses_count(text: str) -> int:
    """Return the number of email addresses in the text."""
    if pd.isna(text):
        return np.nan
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
    return len(re.findall(email_pattern, text))


def unique_syllables_count(text: str, cmudict_dict: dict = None) -> int:
    """Return the number of unique syllables in the text."""
    if pd.isna(text):
        return np.nan
    if not nltk_download('punkt', quiet=True):
        _warn_if_missing_nltk_dependencies('punkt', 'Unique Syllables Count')
        return np.nan
    if cmudict_dict is None:
        if not nltk_download('cmudict', quiet=True):
            _warn_if_missing_nltk_dependencies('cmudict', 'Unique Syllables Count')
            return np.nan
        cmudict_dict = corpus.cmudict.dict()

    text = remove_punctuation(text.lower())
    words = word_tokenize(text)
    syllables = {word: True for word in words if word in cmudict_dict}
    return len(syllables)


def reading_time(text: str) -> int:
    """Return an integer representing time in seconds to read the text.

    The formula is based on Demberg & Keller, 2008 where it is assumed that
    reading a character taken 14.69 milliseconds on average.
    """
    if pd.isna(text):
        return np.nan

    ms_per_char = 14.69
    words = text.split()
    nchars = map(len, words)
    rt_per_word = map(lambda nchar: nchar * ms_per_char, nchars)
    ms_reading_time = sum(list(rt_per_word))
    return round(ms_reading_time / 1000, 2)


def sentences_count(text: str) -> int:
    """Return the number of sentences in the text."""
    if pd.isna(text):
        return np.nan
    if not nltk_download('punkt', quiet=True):
        _warn_if_missing_nltk_dependencies('punkt', 'Sentences Count')
        return np.nan
    return len(_split_to_sentences_with_cache(text))


def average_syllable_length(text: str, cmudict_dict: dict = None) -> float:
    """Return a the average number of syllables per sentences per text sample."""
    if pd.isna(text):
        return np.nan
    if not nltk_download('punkt', quiet=True):
        _warn_if_missing_nltk_dependencies('punkt', 'Average Syllable Length')
        return np.nan
    if cmudict_dict is None:
        if not nltk_download('cmudict', quiet=True):
            _warn_if_missing_nltk_dependencies('cmudict', 'Average Syllable Length')
            return np.nan
        cmudict_dict = corpus.cmudict.dict()
    sentence_count = len(_split_to_sentences_with_cache(text))
    text = remove_punctuation(text.lower())
    words = word_tokenize(text)
    syllable_count = sum([len(cmudict_dict[word]) for word in words if word in cmudict_dict])
    return round(syllable_count / sentence_count, 2)


def _batch_wrapper(text_batch: Sequence[str], func: Callable, **kwargs) -> List[Any]:
    """Wrap the non-batched properties execution with batches API."""
    results = []
    language_property_result = []
    if 'language_property_result' in kwargs:
        language_property_result = kwargs.pop('language_property_result')

    language_property_exists = len(language_property_result) > 0

    for i, text in enumerate(text_batch):
        kwargs['language_property_result'] = language_property_result[i] if language_property_exists else None
        results.append(run_available_kwargs(func, text=text, **kwargs))

    return results


class TextProperty(TypedDict):
    name: str
    method: Callable[..., Sequence[Any]]
    output_type: str


DEFAULT_PROPERTIES: Tuple[TextProperty, ...] = \
    (
        {'name': 'Text Length', 'method': text_length, 'output_type': 'numeric'},
        {'name': 'Average Word Length', 'method': average_word_length, 'output_type': 'numeric'},
        {'name': 'Max Word Length', 'method': max_word_length, 'output_type': 'numeric'},
        {'name': '% Special Characters', 'method': percentage_special_characters, 'output_type': 'numeric'},
        {'name': '% Punctuation', 'method': percentage_punctuation, 'output_type': 'numeric'},
        {'name': 'Language', 'method': language, 'output_type': 'categorical'},
        {'name': 'Sentiment', 'method': sentiment, 'output_type': 'numeric'},
        {'name': 'Subjectivity', 'method': subjectivity, 'output_type': 'numeric'},
        {'name': 'Average Words Per Sentence', 'method': average_words_per_sentence, 'output_type': 'numeric'},
        {'name': 'Reading Ease', 'method': readability_score, 'output_type': 'numeric'},
        {'name': 'Lexical Density', 'method': lexical_density, 'output_type': 'numeric'},
        {'name': 'Toxicity', 'method': toxicity, 'output_type': 'numeric'},
        {'name': 'Fluency', 'method': fluency, 'output_type': 'numeric'},
        {'name': 'Formality', 'method': formality, 'output_type': 'numeric'},
        {'name': 'Unique Noun Count', 'method': unique_noun_count, 'output_type': 'numeric'},
    )

ALL_PROPERTIES: Tuple[TextProperty, ...] = \
    (
        {'name': 'English Text', 'method': english_text, 'output_type': 'categorical'},
        {'name': 'URLs Count', 'method': urls_count, 'output_type': 'numeric'},
        {'name': 'Email Addresses Count', 'method': email_addresses_count, 'output_type': 'numeric'},
        {'name': 'Unique URLs Count', 'method': unique_urls_count, 'output_type': 'numeric'},
        {'name': 'Unique Email Addresses Count', 'method': unique_email_addresses_count, 'output_type': 'numeric'},
        {'name': 'Unique Syllables Count', 'method': unique_syllables_count, 'output_type': 'numeric'},
        {'name': 'Reading Time', 'method': reading_time, 'output_type': 'numeric'},
        {'name': 'Sentences Count', 'method': sentences_count, 'output_type': 'numeric'},
        {'name': 'Average Syllable Length', 'method': average_syllable_length, 'output_type': 'numeric'},
    ) + DEFAULT_PROPERTIES

LONG_RUN_PROPERTIES = ('Toxicity', 'Fluency', 'Formality', 'Unique Noun Count')

BATCH_PROPERTIES = ('Toxicity', 'Fluency', 'Formality')

LARGE_SAMPLE_SIZE = 10_000

ENGLISH_ONLY_PROPERTIES = (
    'Sentiment', 'Subjectivity', 'Toxicity', 'Fluency', 'Formality', 'Reading Ease',
    'Unique Noun Count', 'Unique Syllables Count', 'Sentences Count', 'Average Syllable Length'
)

CMUDICT_PROPERTIES = ('Average Syllable Length', 'Unique Syllables Count', 'Reading Ease')

TEXT_PROPERTIES_DESCRIPTION = {
    'Text Length': 'Number of characters in the text',
    'Average Word Length': 'Average number of characters in a word',
    'Max Word Length': 'Maximum number of characters in a word',
    '% Special Characters': 'Percentage of special characters in the text. Special characters are non-alphanumeric '
                            'unicode characters, excluding whitespaces and any of !\"#$%&\'()*+,-./:;=?\\@.',
    '% Punctuation': 'Percentage of punctuation characters in the text. Punctuation characters are any of '
                     '!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~',
    'Language': 'Language of the text, using the fasttext language detection model',
    'Sentiment': 'Sentiment of the text, calculated using the TextBlob sentiment analysis model.'
                 ' Ranging from -1 (negative) to 1 (positive)',
    'Subjectivity': 'Subjectivity of the text, calculated using the TextBlob sentiment analysis model. Ranging from 0 '
                    '(objective) to 1 (subjective)',
    'Average Words Per Sentence': 'Average number of words per sentence in the text',
    'Reading Ease': 'How easy to read a text sample is, typically ranges from around 0 (hard to read) to around '
                    '100 (very easy). Based on Flesch reading-ease score',
    'Lexical Density': 'Ratio of unique words in the text',
    'Toxicity': 'A measure of how harmful or offensive a text sample is (0 to 1), '
                'uses the SkolkovoInstitute/roberta_toxicity_classifier model',
    'Fluency': 'A measure of the fluency of the text (0 to 1), using the prithivida/parrot_fluency_model'
               ' model from the authors of the Parrot Paraphraser library',
    'Formality': 'The formality / register of the text (0 to 1), using the s-nlp/roberta-base-formality-ranker'
                 ' model by the Skolkovo Institute of Science and Technology',
    'Unique Noun Count': 'Number of unique noun words in the text',
    'URLs Count': 'Number of URLS per text sample',
    'Email Addresses Count': 'Number of email addresses per text sample',
    'Unique URLs Count': 'Number of unique URLS per text sample',
    'English Text': 'Whether the text is in English (1) or not (0)',
    'Unique Email Addresses Count': 'Number of unique email addresses per text sample',
    'Unique Syllables Count': 'Number of unique syllables per text sample',
    'Reading Time': 'Time taken in seconds to read a text sample',
    'Sentences Count': 'Number of sentences per text sample',
    'Average Syllable Length': 'Average number of syllables per sentence per text sample',
}


def _select_properties(
        include_properties: Optional[List[str]] = None,
        ignore_properties: Optional[List[str]] = None,
        include_long_calculation_properties: bool = False,
) -> Sequence[TextProperty]:
    """Select properties to calculate based on provided parameters."""
    if include_properties is not None and ignore_properties is not None:
        raise ValueError('Cannot use properties and ignore_properties parameters together.')
    if include_properties is not None:
        if not is_sequence_not_str(include_properties) \
                and not all(isinstance(prop, str) for prop in include_properties):
            raise DeepchecksValueError('include_properties must be a sequence of strings.')
    if ignore_properties is not None:
        if not is_sequence_not_str(ignore_properties) \
                and not all(isinstance(prop, str) for prop in ignore_properties):
            raise DeepchecksValueError('ignore_properties must be a sequence of strings.')

    include_properties = [prop.lower() for prop in include_properties] if include_properties else None
    ignore_properties = [prop.lower() for prop in ignore_properties] if ignore_properties else None

    if include_properties is not None:
        properties = [prop for prop in ALL_PROPERTIES if
                      prop['name'].lower() in include_properties]  # pylint: disable=unsupported-membership-test
        if len(properties) < len(include_properties):
            not_found_properties = sorted(set(include_properties) - set(prop['name'].lower() for prop in properties))
            raise DeepchecksValueError('include_properties contains properties that were not found: '
                                       f'{not_found_properties}.')
    elif ignore_properties is not None:
        properties = [prop for prop in DEFAULT_PROPERTIES if
                      prop['name'].lower() not in ignore_properties]  # pylint: disable=unsupported-membership-test
        if len(properties) + len(ignore_properties) != len(DEFAULT_PROPERTIES):
            default_property_names = [prop['name'].lower() for prop in DEFAULT_PROPERTIES]
            not_found_properties = [prop for prop in list(ignore_properties) if prop not in default_property_names]
            raise DeepchecksValueError('ignore_properties contains properties that were not found: '
                                       f'{not_found_properties}.')
    else:
        properties = DEFAULT_PROPERTIES

    # include_long_calculation_properties is only applicable when include_properties is None
    if include_properties is None and not include_long_calculation_properties:
        return [
            prop for prop in properties
            if prop['name'] not in LONG_RUN_PROPERTIES
        ]

    return properties


def calculate_builtin_properties(
        raw_text: Sequence[str],
        include_properties: Optional[List[str]] = None,
        ignore_properties: Optional[List[str]] = None,
        include_long_calculation_properties: bool = False,
        ignore_non_english_samples_for_english_properties: bool = True,
        device: Optional[str] = None,
        models_storage: Union[pathlib.Path, str, None] = None,
        batch_size: Optional[int] = 16,
        cache_models: bool = False,
        use_onnx_models: bool = True,
) -> Tuple[Dict[str, List[float]], Dict[str, str]]:
    """Calculate properties on provided text samples.

    Parameters
    ----------
    raw_text : Sequence[str]
        The text to calculate the properties for.
    include_properties : List[str], default None
        The properties to calculate. If None, all default properties will be calculated. Cannot be used
        together with ignore_properties parameter. Available properties are:
        ['Text Length', 'Average Word Length', 'Max Word Length', '% Special Characters', '% Punctuation', 'Language',
        'Sentiment', 'Subjectivity', 'Toxicity', 'Fluency', 'Formality', 'Lexical Density', 'Unique Noun Count',
        'Reading Ease', 'Average Words Per Sentence', 'URLs Count', Unique URLs Count', 'Email Address Count',
        'Unique Email Address Count', 'Unique Syllables Count', 'Reading Time', 'Sentences Count',
        'Average Syllable Length']
        List of default properties are: ['Text Length', 'Average Word Length', 'Max Word Length',
        '% Special Characters', '% Punctuation', 'Language', 'Sentiment', 'Subjectivity', 'Toxicity', 'Fluency',
        'Formality', 'Lexical Density', 'Unique Noun Count', 'Reading Ease', 'Average Words Per Sentence']
        To calculate all the default properties, the include_properties and ignore_properties parameters should
        be None. If you pass either include_properties or ignore_properties then only the properties specified
        in the list will be calculated or ignored.
        Note that the properties ['Toxicity', 'Fluency', 'Formality', 'Language', 'Unique Noun Count'] may
        take a long time to calculate. If include_long_calculation_properties is False, these properties will be
        ignored, even if they are in the include_properties parameter.
    ignore_properties : List[str], default None
        The properties to ignore from the list of default properties. If None, no properties will be ignored and
        all the default properties will be calculated. Cannot be used together with include_properties parameter.
    include_long_calculation_properties : bool, default False
        Whether to include properties that may take a long time to calculate. If False, these properties will be
        ignored, unless they are specified in the include_properties parameter explicitly.
    ignore_non_english_samples_for_english_properties : bool, default True
        Whether to ignore samples that are not in English when calculating English properties. If False, samples
        that are not in English will be calculated as well. This parameter is ignored when calculating non-English
        properties.
        English-Only properties WILL NOT work properly on non-English samples, and this parameter should be used
        only when you are sure that all the samples are in English.
    device : Optional[str], default None
        The device to use for the calculation. If None, the default device will be used. For onnx based models it is
        recommended to set device to None for optimized performance.
    models_storage : Union[str, pathlib.Path, None], default None
        A directory to store the models.
        If not provided, models will be stored in `DEEPCHECKS_LIB_PATH/nlp/.nlp-models`.
        Also, if a folder already contains relevant resources they are not re-downloaded.
    batch_size : int, default 8
        The batch size.
    cache_models : bool, default False
        If True, will store the models in device RAM memory. This will speed up the calculation for future calls.
    use_onnx_models : bool, default True
        If True, will use onnx gpu optimized models for the calculation. Requires the optimum[onnxruntime-gpu] library
        to be installed as well as the availability of GPU.

    Returns
    -------
    Dict[str, List[float]]
        A dictionary with the property name as key and a list of the property values for each text as value.
    Dict[str, str]
        A dictionary with the property name as key and the property's type as value.
    """
    use_onnx_models = _validate_onnx_model_availability(use_onnx_models, device)
    text_properties = _select_properties(
        include_properties=include_properties,
        ignore_properties=ignore_properties,
        include_long_calculation_properties=include_long_calculation_properties
    )

    properties_types = {
        it['name']: it['output_type']
        for it in text_properties
    }
    _warn_long_compute(device, properties_types, len(raw_text), use_onnx_models)

    kwargs = dict(device=device, models_storage=models_storage)
    calculated_properties = {k: [] for k in properties_types.keys()}

    # Prepare kwargs for properties that require outside resources:
    kwargs['fasttext_model'] = get_fasttext_model(models_storage=models_storage, use_cache=cache_models)

    properties_requiring_cmudict = list(set(CMUDICT_PROPERTIES) & set(properties_types.keys()))
    if properties_requiring_cmudict:
        if not nltk_download('cmudict', quiet=True):
            _warn_if_missing_nltk_dependencies('cmudict', format_list(properties_requiring_cmudict))
            for prop in properties_requiring_cmudict:
                calculated_properties[prop] = [np.nan] * len(raw_text)
        kwargs['cmudict_dict'] = get_cmudict_dict(use_cache=cache_models)

    if 'Toxicity' in properties_types and 'toxicity_classifier' not in kwargs:
        model_name = TOXICITY_MODEL_NAME_ONNX if use_onnx_models else TOXICITY_MODEL_NAME
        kwargs['toxicity_classifier'] = get_transformer_pipeline(
            property_name='toxicity', model_name=model_name, device=device,
            models_storage=models_storage, use_cache=cache_models, use_onnx_model=use_onnx_models)

    if 'Formality' in properties_types and 'formality_classifier' not in kwargs:
        model_name = FORMALITY_MODEL_NAME_ONNX if use_onnx_models else FORMALITY_MODEL_NAME
        kwargs['formality_classifier'] = get_transformer_pipeline(
            property_name='formality', model_name=model_name, device=device,
            models_storage=models_storage, use_cache=cache_models, use_onnx_model=use_onnx_models)

    if 'Fluency' in properties_types and 'fluency_classifier' not in kwargs:
        model_name = FLUENCY_MODEL_NAME_ONNX if use_onnx_models else FLUENCY_MODEL_NAME
        kwargs['fluency_classifier'] = get_transformer_pipeline(
            property_name='fluency', model_name=model_name, device=device,
            models_storage=models_storage, use_cache=cache_models, use_onnx_model=use_onnx_models)

    # Remove language property from the list of properties to calculate as it will be calculated separately:
    text_properties = [prop for prop in text_properties if prop['name'] != 'Language']

    warning_message = (
        'Failed to calculate property {0}. '
        'Dependencies required by property are not installed. '
        'Error:\n{1}'
    )
    import_warnings = set()

    # Calculate all properties for a specific batch than continue to the next batch
    for i in tqdm(range(0, len(raw_text), batch_size)):
        batch = raw_text[i:i + batch_size]
        batch_properties = defaultdict(list)

        # filtering out empty sequences
        nan_indices = {i for i, seq in enumerate(batch) if pd.isna(seq) is True}
        filtered_sequences = [e for i, e in enumerate(batch) if i not in nan_indices]

        samples_language = _batch_wrapper(text_batch=filtered_sequences, func=language, **kwargs)
        if 'Language' in properties_types:
            batch_properties['Language'].extend(samples_language)
            calculated_properties['Language'].extend(samples_language)
        kwargs['language_property_result'] = samples_language  # Pass the language property to other properties
        kwargs['batch_size'] = batch_size

        non_english_indices = set()
        if ignore_non_english_samples_for_english_properties:
            non_english_indices = {i for i, (seq, lang) in enumerate(zip(filtered_sequences, samples_language))
                                   if lang != 'en'}

        for prop in text_properties:
            if prop['name'] in import_warnings:  # Skip properties that failed to import:
                batch_properties[prop['name']].extend([np.nan] * len(batch))
                continue

            sequences_to_use = list(filtered_sequences)
            if prop['name'] in ENGLISH_ONLY_PROPERTIES and ignore_non_english_samples_for_english_properties:
                sequences_to_use = [e for i, e in enumerate(sequences_to_use) if i not in non_english_indices]
            try:
                if prop['name'] in BATCH_PROPERTIES:
                    value = run_available_kwargs(text_batch=sequences_to_use, func=prop['method'], **kwargs)
                else:
                    value = _batch_wrapper(text_batch=sequences_to_use, func=prop['method'], **kwargs)
                batch_properties[prop['name']].extend(value)
            except ImportError as e:
                warnings.warn(warning_message.format(prop['name'], str(e)))
                batch_properties[prop['name']].extend([np.nan] * len(batch))
                import_warnings.add(prop['name'])
                continue

            # Fill in nan values for samples that were filtered out:
            result_index = 0
            for index, seq in enumerate(batch):
                if index in nan_indices or (index in non_english_indices and
                                            ignore_non_english_samples_for_english_properties and
                                            prop['name'] in ENGLISH_ONLY_PROPERTIES):
                    calculated_properties[prop['name']].append(np.nan)
                else:
                    calculated_properties[prop['name']].append(batch_properties[prop['name']][result_index])
                    result_index += 1

        # Clear property caches:
        textblob_cache.clear()
        words_cache.clear()
        sentences_cache.clear()

    if not calculated_properties:
        raise RuntimeError('Failed to calculate any of the properties.')

    properties_types = {
        k: v
        for k, v in properties_types.items()
        if k in calculated_properties
    }

    return calculated_properties, properties_types


def _warn_long_compute(device, properties_types, n_samples, use_onnx_models):
    heavy_properties = [prop for prop in properties_types.keys() if prop in LONG_RUN_PROPERTIES]
    if len(heavy_properties) and n_samples > LARGE_SAMPLE_SIZE:
        warning_message = (
            f'Calculating the properties {heavy_properties} on a large dataset may take a long time. '
            'Consider using a smaller sample size or running this code on better hardware.'
        )
        if device == 'cpu' or (device is None and not use_onnx_models):
            warning_message += ' Consider using a GPU or a similar device to run these properties.'
        warnings.warn(warning_message, UserWarning)


def _validate_onnx_model_availability(use_onnx_models: bool, device: Optional[str]):
    if not use_onnx_models:
        return False
    if find_spec('optimum') is None or find_spec('onnxruntime') is None:
        warnings.warn('Onnx models require the optimum[onnxruntime-gpu] library to be installed. '
                      'Calculating using the default models.')
        return False
    if not torch.cuda.is_available():
        warnings.warn('GPU is required for the onnx models. Calculating using the default models.')
        return False
    if device is not None and device.lower() == 'cpu':
        warnings.warn('Onnx models are not supported on device CPU. Calculating using the default models.')
        return False
    return True


def get_builtin_properties_types():
    """
    Get the names of all the available builtin properties.

    Returns
    -------
    Dict[str, str]
        A dictionary with the property name as key and the property's type as value.
    """
    return {
        prop['name']: prop['output_type']
        for prop in ALL_PROPERTIES
    }
