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
import gc
import importlib
import pathlib
import re
import string
import warnings
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import requests
import textblob
from nltk import corpus
from nltk import download as nltk_download
from nltk import sent_tokenize, word_tokenize
from tqdm import tqdm
from typing_extensions import TypedDict

from deepchecks.core.errors import DeepchecksValueError
from deepchecks.nlp.utils.text import cut_string, hash_text, normalize_text, remove_punctuation
from deepchecks.utils.function import run_available_kwargs
from deepchecks.utils.strings import format_list

__all__ = ['calculate_builtin_properties', 'get_builtin_properties_types']

from deepchecks.utils.validation import is_sequence_not_str

MODELS_STORAGE = pathlib.Path(__file__).absolute().parent / '.nlp-models'
FASTTEXT_LANG_MODEL = 'https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin'
DEFAULT_SENTENCE_SAMPLE_SIZE = 300
MAX_CHARS = 512  # Bert accepts max of 512 tokens, so without counting tokens we go for the lower bound.
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


def _import_optional_property_dependency(
        module: str,
        property_name: str,
        package_name: Optional[str] = None,
        error_template: Optional[str] = None
):
    try:
        lib = importlib.import_module(module)
    except ImportError as error:
        package_name = package_name or module.split('.', maxsplit=1)[0]
        error_template = error_template or (
            'property {property_name} requires the {package_name} python package. '
            'To get it, run:\n'
            '>> pip install {package_name}\n\n'
            'You may install dependencies for all text properties by running:\n'
            '>> pip install deepchecks[nlp-properties]\n'
        )
        raise ImportError(error_template.format(
            property_name=property_name,
            package_name=package_name
        )) from error
    else:
        return lib


def _warn_if_missing_nltk_dependencies(dependency: str, property_name: str):
    """Warn if NLTK dependency is missing."""
    warnings.warn(f'NLTK {dependency} not found, {property_name} cannot be calculated.'
                  ' Please check your internet connection.', UserWarning)


def get_create_model_storage(models_storage: Union[pathlib.Path, str, None] = None):
    """Get the models storage directory and create it if needed."""
    if models_storage is None:
        models_storage = MODELS_STORAGE
    else:
        if isinstance(models_storage, str):
            models_storage = pathlib.Path(models_storage)
        if not isinstance(models_storage, pathlib.Path):
            raise ValueError(
                f'Unexpected type of the "models_storage" parameter - {type(models_storage)}'
            )
        if not models_storage.exists():
            models_storage.mkdir(parents=True)
        if not models_storage.is_dir():
            raise ValueError('"model_storage" expected to be a directory')

    return models_storage


def get_transformer_model(
        property_name: str,
        model_name: str,
        device: Optional[str] = None,
        quantize_model: bool = False,
        models_storage: Union[pathlib.Path, str, None] = None
):
    """Get the transformer model and decide if to use optimum.onnxruntime.

    optimum.onnxruntime is used to optimize running times on CPU.
    """
    models_storage = get_create_model_storage(models_storage)

    if device not in (None, 'cpu'):
        transformers = _import_optional_property_dependency('transformers', property_name=property_name)
        # TODO: quantize if 'quantize_model' is True
        return transformers.AutoModelForSequenceClassification.from_pretrained(
            model_name,
            cache_dir=models_storage,
            device_map=device
        )

    onnx = _import_optional_property_dependency(
        'optimum.onnxruntime',
        property_name=property_name,
        error_template=(
            f'The device was set to {device} while computing the {property_name} property,'
            'in which case deepchecks resorts to accelerating the inference by using optimum,'
            'bit it is not installed. Either:\n'
            '\t- Set the device according to your hardware;\n'
            '\t- Install optimum by running "pip install optimum";\n'
            '\t- Install all dependencies needed for text properties by running '
            '"pip install deepchecks[nlp-properties]";\n'
        )
    )

    if quantize_model is False:
        model_path = models_storage / 'onnx' / model_name

        if model_path.exists():
            return onnx.ORTModelForSequenceClassification.from_pretrained(model_path, device_map=device)

        model = onnx.ORTModelForSequenceClassification.from_pretrained(
            model_name,
            export=True,
            cache_dir=models_storage,
            device_map=device
        )
        # NOTE:
        # 'optimum', after exporting/converting a model to the ONNX format,
        # does not store it onto disk we need to save it now to not reconvert
        # it each time
        model.save_pretrained(model_path)
        return model

    model_path = models_storage / 'onnx' / 'quantized' / model_name

    if model_path.exists():
        return onnx.ORTModelForSequenceClassification.from_pretrained(model_path, device_map=device)

    not_quantized_model = get_transformer_model(
        property_name,
        model_name,
        device,
        quantize_model=False,
        models_storage=models_storage
    )

    quantizer = onnx.ORTQuantizer.from_pretrained(not_quantized_model, device_map=device)

    quantizer.quantize(
        save_dir=model_path,
        # TODO: make it possible to provide a config as a parameter
        quantization_config=onnx.configuration.AutoQuantizationConfig.avx512_vnni(
            is_static=False,
            per_channel=False
        )
    )
    return onnx.ORTModelForSequenceClassification.from_pretrained(model_path, device_map=device)


def get_transformer_pipeline(
        property_name: str,
        model_name: str,
        device: Optional[str] = None,
        models_storage: Union[pathlib.Path, str, None] = None
):
    """Return a transformers pipeline for the given model name."""
    transformers = _import_optional_property_dependency('transformers', property_name=property_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, device_map=device)
    model = get_transformer_model(
        property_name=property_name,
        model_name=model_name,
        device=device,
        models_storage=models_storage
    )
    return transformers.pipeline(
        'text-classification',
        model=model,
        tokenizer=tokenizer,
        device=device
    )


def text_length(text: str) -> int:
    """Return text length."""
    return len(text)


def average_word_length(text: str) -> float:
    """Return average word length."""
    words = _split_to_words_with_cache(text)
    return np.mean([len(word) for word in words]) if words else 0


def percentage_special_characters(text: str) -> float:
    """Return percentage of special characters (as float between 0 and 1)."""
    return len([c for c in text if c in string.punctuation]) / len(text) if len(text) != 0 else 0


def max_word_length(text: str) -> int:
    """Return max word length."""
    words = _split_to_words_with_cache(text)
    return max(len(w) for w in words) if words else 0


def _get_fasttext_model(models_storage: Union[pathlib.Path, str, None] = None):
    """Return fasttext model."""
    fasttext = _import_optional_property_dependency(module='fasttext', property_name='language')

    model_name = FASTTEXT_LANG_MODEL.rsplit('/', maxsplit=1)[-1]
    model_path = get_create_model_storage(models_storage)
    model_path = model_path / 'fasttext'

    if not model_path.exists():
        model_path.mkdir(parents=True)

    model_path = model_path / model_name

    # Save the model to a file
    if not model_path.exists():
        response = requests.get(FASTTEXT_LANG_MODEL, timeout=240)
        if response.status_code != 200:
            raise RuntimeError('Failed to donwload fasttext model')
        model_path.write_bytes(response.content)

    # This weird code is to suppress a warning from fasttext about a deprecated function
    try:
        fasttext.FastText.eprint = lambda *args, **kwargs: None
        fasttext_model = fasttext.load_model(str(model_path))
    except Exception as exp:
        raise exp

    return fasttext_model


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
        fasttext_model = _get_fasttext_model()

    # Predictions are the first prediction (k=1), only if the probability is above the threshold
    prediction = fasttext_model.predict(text.replace('\n', ' '), k=1, threshold=lang_certainty_threshold)[0]
    # label is empty for detection below threshold:
    language_code = prediction[0].replace('__label__', '') if prediction else None
    return language_code


def is_english(
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


def _predict(text_batch: Sequence[str], classifier, kind: str, batch_size: int) -> Sequence[float]:
    """Return prediction of huggingface Pipeline classifier."""
    # TODO: make this way smarter, and not just a hack. Count tokens, for a start. Then not just sample sentences.
    # If text is longer than classifier context window, sample it:
    text_list_to_predict = []
    reduced_batch_size = batch_size  # Initialize the reduced batch size

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
            if reduced_batch_size == 1:
                results = []
                for text in text_list_to_predict:
                    try:
                        v = classifier(text)
                        if not v:
                            results.append(np.nan)
                        elif kind == 'toxicity':
                            results.append(v['score'])
                        elif kind == 'fluency':
                            results.append(v['score'] if v['label'] == 'LABEL_1' else 1 - v['score'])
                        elif kind == 'formality':
                            results.append(v['score'] if v['label'] == 'formal' else 1 - v['score'])
                        else:
                            raise ValueError('Unsupported value for "kind" parameter')
                    except Exception:  # pylint: disable=broad-except
                        results.append(np.nan)
                return results  # Return the results if prediction is successful

            v_list = classifier(text_list_to_predict, batch_size=reduced_batch_size)
            results = []

            for v in v_list:
                if not v:
                    results.append(np.nan)
                elif kind == 'toxicity':
                    results.append(v['score'])
                elif kind == 'fluency':
                    results.append(v['score'] if v['label'] == 'LABEL_1' else 1 - v['score'])
                elif kind == 'formality':
                    results.append(v['score'] if v['label'] == 'formal' else 1 - v['score'])
                else:
                    raise ValueError('Unsupported value for "kind" parameter')

            return results  # Return the results if prediction is successful

        except Exception:  # pylint: disable=broad-except
            reduced_batch_size = max(reduced_batch_size // 2, 1)  # Reduce the batch size by half
            text_list_to_predict = []  # Clear the list of texts to predict for retry

    return [np.nan] * batch_size  # Prediction failed, return NaN values for the original batch size


TOXICITY_MODEL_NAME = 'unitary/toxic-bert'
FLUENCY_MODEL_NAME = 'prithivida/parrot_fluency_model'
FORMALITY_MODEL_NAME = 's-nlp/roberta-base-formality-ranker'


def toxicity(
        text_batch: Sequence[str],
        batch_size: int = 1,
        device: Optional[str] = None,
        models_storage: Union[pathlib.Path, str, None] = None,
        toxicity_classifier: Optional[object] = None
) -> Sequence[float]:
    """Return float representing toxicity."""
    if toxicity_classifier is None:
        toxicity_classifier = get_transformer_pipeline(
            property_name='toxicity', model_name=TOXICITY_MODEL_NAME, device=device, models_storage=models_storage)
    return _predict(text_batch, toxicity_classifier, 'toxicity', batch_size)


def fluency(
        text_batch: Sequence[str],
        batch_size: int = 1,
        device: Optional[str] = None,
        models_storage: Union[pathlib.Path, str, None] = None,
        fluency_classifier: Optional[object] = None
) -> Sequence[float]:
    """Return float representing fluency."""
    if fluency_classifier is None:
        fluency_classifier = get_transformer_pipeline(
            property_name='fluency', model_name=FLUENCY_MODEL_NAME, device=device, models_storage=models_storage)
    return _predict(text_batch, fluency_classifier, 'fluency', batch_size)


def formality(
        text_batch: Sequence[str],
        batch_size: int = 1,
        device: Optional[str] = None,
        models_storage: Union[pathlib.Path, str, None] = None,
        formality_classifier: Optional[object] = None
) -> Sequence[float]:
    """Return float representing formality."""
    if formality_classifier is None:
        formality_classifier = get_transformer_pipeline(
            property_name='formality', model_name=FORMALITY_MODEL_NAME, device=device, models_storage=models_storage)
    return _predict(text_batch, formality_classifier, 'formality', batch_size)


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
    return round(total_unique_words * 100 / len(all_words), 2)


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
            _warn_if_missing_nltk_dependencies('cmudict', 'Readability Score')
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
    for text in text_batch:
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
        {'name': 'Language', 'method': language, 'output_type': 'categorical'},
        {'name': 'Sentiment', 'method': sentiment, 'output_type': 'numeric'},
        {'name': 'Subjectivity', 'method': subjectivity, 'output_type': 'numeric'},
        {'name': 'Average Words Per Sentence', 'method': average_words_per_sentence, 'output_type': 'numeric'},
        {'name': 'Readability Score', 'method': readability_score, 'output_type': 'numeric'},
        {'name': 'Lexical Density', 'method': lexical_density, 'output_type': 'numeric'},
        {'name': 'Toxicity', 'method': toxicity, 'output_type': 'numeric'},
        {'name': 'Fluency', 'method': fluency, 'output_type': 'numeric'},
        {'name': 'Formality', 'method': formality, 'output_type': 'numeric'},
        {'name': 'Unique Noun Count', 'method': unique_noun_count, 'output_type': 'numeric'},
    )

ALL_PROPERTIES: Tuple[TextProperty, ...] = \
    (
        {'name': 'Is English', 'method': is_english, 'output_type': 'categorical'},
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
    'Sentiment', 'Subjectivity', 'Toxicity', 'Fluency', 'Formality', 'Readability Score',
    'Unique Noun Count', 'Unique Syllables Count', 'Sentences Count', 'Average Syllable Length'
)

CMUDICT_PROPERTIES = ('Average Syllable Length', 'Unique Syllables Count', 'Readability Score')

TEXT_PROPERTIES_DESCRIPTION = {
    'Text Length': 'Number of characters in the text',
    'Average Word Length': 'Average number of characters in a word',
    'Max Word Length': 'Maximum number of characters in a word',
    '% Special Characters': 'Percentage of special characters in the text',
    'Language': 'Language of the text, using the fasttext language detection model',
    'Sentiment': 'Sentiment of the text, calculated using the TextBlob sentiment analysis model',
    'Subjectivity': 'Subjectivity of the text, calculated using the TextBlob sentiment analysis model',
    'Average Words Per Sentence': 'Average number of words per sentence in the text',
    'Readability Score': 'A score calculated based on Flesch reading-ease per text sample',
    'Lexical Density': 'Percentage of unique words in the text',
    'Toxicity': 'Toxicity score using unitary/toxic-bert HuggingFace model',
    'Fluency': 'Fluency score using prithivida/parrot_fluency_model HuggingFace model',
    'Formality': 'Formality score using s-nlp/roberta-base-formality-ranker HuggingFace model',
    'Unique Noun Count': 'Number of unique noun words in the text',
    'URLs Count': 'Number of URLS per text sample',
    'Email Addresses Count': 'Number of email addresses per text sample',
    'Unique URLs Count': 'Number of unique URLS per text sample',
    'Unique Email Addresses Count': 'Number of unique email addresses per text sample',
    'Unique Syllables Count': 'Number of unique syllables per text sample',
    'Reading Time': 'Time taken in seconds to read a text sample',
    'Sentences Count': 'Number of sentences per text sample',
    'Average Syllable Length': 'Average number of syllables per sentence per text sample',
}


def _select_properties(
        *,
        n_of_samples: int,
        include_properties: Optional[List[str]] = None,
        ignore_properties: Optional[List[str]] = None,
        include_long_calculation_properties: bool = False,
        device: Optional[str] = None,
) -> Sequence[TextProperty]:
    """Select properties."""
    all_properties = ALL_PROPERTIES
    default_properties = DEFAULT_PROPERTIES

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
        properties = [prop for prop in all_properties if prop['name'].lower() in include_properties]
        if len(properties) < len(include_properties):
            not_found_properties = sorted(set(include_properties) - set(prop['name'].lower() for prop in properties))
            raise DeepchecksValueError('include_properties contains properties that were not found: '
                                       f'{not_found_properties}.')
    elif ignore_properties is not None:
        properties = [prop for prop in default_properties if prop['name'].lower() not in ignore_properties]
        if len(properties) + len(ignore_properties) != len(default_properties):
            not_found_properties = \
                [prop for prop in ignore_properties if prop not in [prop['name'] for prop in default_properties]]
            raise DeepchecksValueError('ignore_properties contains properties that were not found: '
                                       f'{not_found_properties}.')
    else:
        properties = default_properties

    # include_long_calculation_properties is only applicable when include_properties is None
    if not include_long_calculation_properties and include_properties is None:
        return [
            prop for prop in properties
            if prop['name'] not in LONG_RUN_PROPERTIES
        ]

    heavy_properties = [
        prop for prop in properties
        if prop['name'] in LONG_RUN_PROPERTIES
    ]

    if heavy_properties and n_of_samples > LARGE_SAMPLE_SIZE:
        h_prop_names = [
            prop['name']
            for prop in heavy_properties
        ]
        warning_message = (
            f'Calculating the properties {h_prop_names} on a large dataset may take a long time. '
            'Consider using a smaller sample size or running this code on better hardware.'
        )
        if device is None or device == 'cpu':
            warning_message += ' Consider using a GPU or a similar device to run these properties.'
        warnings.warn(warning_message, UserWarning)

    return properties


def calculate_builtin_properties(
        raw_text: Sequence[str],
        include_properties: Optional[List[str]] = None,
        ignore_properties: Optional[List[str]] = None,
        include_long_calculation_properties: bool = False,
        ignore_non_english_samples_for_english_properties: bool = True,
        device: Optional[str] = None,
        models_storage: Union[pathlib.Path, str, None] = None,
        batch_size: Optional[int] = 1
) -> Tuple[Dict[str, List[float]], Dict[str, str]]:
    """Calculate properties on provided text samples.

    Parameters
    ----------
    raw_text : Sequence[str]
        The text to calculate the properties for.
    include_properties : List[str], default None
        The properties to calculate. If None, all default properties will be calculated. Cannot be used
        together with ignore_properties parameter. Available properties are:
        ['Text Length', 'Average Word Length', 'Max Word Length', '% Special Characters', 'Language',
        'Sentiment', 'Subjectivity', 'Toxicity', 'Fluency', 'Formality', 'Lexical Density', 'Unique Noun Count',
        'Readability Score', 'Average Words Per Sentence', 'URLs Count', Unique URLs Count', 'Email Address Count',
        'Unique Email Address Count', 'Unique Syllables Count', 'Reading Time', 'Sentences Count',
        'Average Syllable Length']
        List of default properties are: ['Text Length', 'Average Word Length', 'Max Word Length',
        '% Special Characters', 'Language', 'Sentiment', 'Subjectivity', 'Toxicity', 'Fluency', 'Formality',
        'Lexical Density', 'Unique Noun Count', 'Readability Score', 'Average Words Per Sentence']
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
    device : int, default None
        The device to use for the calculation. If None, the default device will be used.
    models_storage : Union[str, pathlib.Path, None], default None
        A directory to store the models.
        If not provided, models will be stored in `DEEPCHECKS_LIB_PATH/nlp/.nlp-models`.
        Also, if a folder already contains relevant resources they are not re-downloaded.

    Returns
    -------
    Dict[str, List[float]]
        A dictionary with the property name as key and a list of the property values for each text as value.
    Dict[str, str]
        A dictionary with the property name as key and the property's type as value.
    """
    text_properties = _select_properties(
        include_properties=include_properties,
        ignore_properties=ignore_properties,
        device=device,
        include_long_calculation_properties=include_long_calculation_properties,
        n_of_samples=len(raw_text)
    )
    properties_types = {
        it['name']: it['output_type']
        for it in text_properties
    }

    kwargs = dict(device=device, models_storage=models_storage)
    english_properties_names = set(ENGLISH_ONLY_PROPERTIES)
    text_properties_names = [it['name'] for it in text_properties]
    calculated_properties = {k: [] for k in text_properties_names}

    # Prepare kwargs for properties that require outside resources:
    if 'fasttext_model' not in kwargs:
        kwargs['fasttext_model'] = _get_fasttext_model(models_storage=models_storage)

    if 'cmudict_dict' not in kwargs:
        properties_requiring_cmudict = list(set(CMUDICT_PROPERTIES) & set(text_properties_names))
        if properties_requiring_cmudict:
            if not nltk_download('cmudict', quiet=True):
                _warn_if_missing_nltk_dependencies('cmudict', format_list(properties_requiring_cmudict))
                for prop in properties_requiring_cmudict:
                    calculated_properties[prop] = [np.nan] * len(raw_text)
            cmudict_dict = corpus.cmudict.dict()
            kwargs['cmudict_dict'] = cmudict_dict

    if 'Toxicity' in text_properties_names and 'toxicity_classifier' not in kwargs:
        kwargs['toxicity_classifier'] = get_transformer_pipeline(
            property_name='toxicity', model_name=TOXICITY_MODEL_NAME, device=device, models_storage=models_storage)

    if 'Formality' in text_properties_names and 'formality_classifier' not in kwargs:
        kwargs['formality_classifier'] = get_transformer_pipeline(
            property_name='formality', model_name=FORMALITY_MODEL_NAME, device=device, models_storage=models_storage)

    if 'Fluency' in text_properties_names and 'fluency_classifier' not in kwargs:
        kwargs['fluency_classifier'] = get_transformer_pipeline(
            property_name='fluency', model_name=FLUENCY_MODEL_NAME, device=device, models_storage=models_storage)

    is_language_property_requested = 'Language' in [prop['name'] for prop in text_properties]
    # Remove language property from the list of properties to calculate as it will be calculated separately:
    if is_language_property_requested:
        text_properties = [prop for prop in text_properties if prop['name'] != 'Language']

    warning_message = (
        'Failed to calculate property {0}. '
        'Dependencies required by property are not installed. '
        'Error:\n{1}'
    )
    import_warnings = set()

    for i in tqdm(range(0, len(raw_text), batch_size)):
        batch = raw_text[i:i + batch_size]
        batch_properties = defaultdict(list)

        # filtering out empty sequences
        filtered_sequences = [seq for seq in batch if pd.isna(seq) is False]

        samples_language = _batch_wrapper(text_batch=filtered_sequences, func=language, **kwargs)
        if is_language_property_requested:
            batch_properties['Language'].extend(samples_language)
        kwargs['language_property_result'] = samples_language  # Pass the language property to other properties

        for prop in text_properties:
            if prop['name'] in import_warnings:  # Skip properties that failed to import:
                batch_properties[prop['name']].extend([np.nan] * len(batch))
            else:
                if prop['name'] in english_properties_names \
                        and ignore_non_english_samples_for_english_properties is True:
                    filtered_sequences = \
                        [seq for seq, lang in zip(filtered_sequences, samples_language) if lang == 'en']
                kwargs['batch_size'] = batch_size
                try:
                    if prop['name'] in BATCH_PROPERTIES:
                        value = run_available_kwargs(func=prop['method'], text_batch=filtered_sequences, **kwargs)
                    else:
                        value = _batch_wrapper(text_batch=filtered_sequences, func=prop['method'], **kwargs)
                    batch_properties[prop['name']].extend(value)
                except ImportError as e:
                    warnings.warn(warning_message.format(prop['name'], str(e)))
                    batch_properties[prop['name']].extend([np.nan] * len(batch))
                    import_warnings.add(prop['name'])

            calculated_properties[prop['name']].extend([prop if seq is not None else np.nan
                                                        for seq, prop in zip(batch, batch_properties[prop['name']])])

        # Clear property caches:
        textblob_cache.clear()
        words_cache.clear()
        sentences_cache.clear()

    # Clean all remaining RAM:
    gc.collect()

    if not calculated_properties:
        raise RuntimeError('Failed to calculate any of the properties.')

    properties_types = {
        k: v
        for k, v in properties_types.items()
        if k in calculated_properties
    }

    return calculated_properties, properties_types


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
