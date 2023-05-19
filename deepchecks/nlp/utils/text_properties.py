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
import importlib
import pathlib
import re
import string
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import requests
import textblob
from nltk import corpus
from nltk import download as nltk_download
from nltk import sent_tokenize, word_tokenize
from typing_extensions import TypedDict

from deepchecks.nlp.utils.text import remove_punctuation
from deepchecks.utils.function import run_available_kwargs
from deepchecks.utils.ipython import create_progress_bar

__all__ = ['calculate_builtin_properties']


MODELS_STORAGE = pathlib.Path(__file__).absolute().parent / '.nlp-models'
FASTTEXT_LANG_MODEL = 'https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin'


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


def get_creat_model_storage(models_storage: Union[pathlib.Path, str, None] = None):
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
    models_storage = get_creat_model_storage(models_storage)

    if device not in (None, 'cpu'):
        transformers = _import_optional_property_dependency('transformers', property_name=property_name)
        # TODO: quantize if 'quantize_model' is True
        return transformers.AutoModelForSequenceClassification.from_pretrained(
            model_name,
            cache_dir=models_storage
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
            return onnx.ORTModelForSequenceClassification.from_pretrained(model_path)

        model = onnx.ORTModelForSequenceClassification.from_pretrained(
            model_name,
            export=True,
            cache_dir=models_storage
        )
        # NOTE:
        # 'optimum', after exporting/converting a model to the ONNX format,
        # does not store it onto disk we need to save it now to not reconvert
        # it each time
        model.save_pretrained(model_path)
        return model

    model_path = models_storage / 'onnx' / 'quantized' / model_name

    if model_path.exists():
        return onnx.ORTModelForSequenceClassification.from_pretrained(model_path)

    not_quantized_model = get_transformer_model(
        property_name,
        model_name,
        device,
        quantize_model=False,
        models_storage=models_storage
    )

    quantizer = onnx.ORTQuantizer.from_pretrained(not_quantized_model)

    quantizer.quantize(
        save_dir=model_path,
        # TODO: make it possible to provide a config as a parameter
        quantization_config=onnx.configuration.AutoQuantizationConfig.avx512_vnni(
            is_static=False,
            per_channel=False
        )
    )
    return onnx.ORTModelForSequenceClassification.from_pretrained(model_path)


def get_transformer_pipeline(
    property_name: str,
    model_name: str,
    device: Optional[str] = None,
    models_storage: Union[pathlib.Path, str, None] = None
):
    """Return a transformers pipeline for the given model name."""
    transformers = _import_optional_property_dependency('transformers', property_name=property_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
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


def text_length(raw_text: Sequence[str]) -> List[int]:
    """Return list of integers of text lengths."""
    return [len(text) for text in raw_text]


def word_length(raw_text: Sequence[str]) -> List[int]:  # Not yet used as returns list per sample and not number
    """Return list of integers of word lengths."""
    return [len(word) for text in raw_text for word in text.split()]


def average_word_length(raw_text: Sequence[str]) -> List[float]:
    """Return list of floats of average word length."""
    return [np.mean([len(word) for word in text.split()]) for text in raw_text]


def percentage_special_characters(raw_text: Sequence[str]) -> List[float]:
    """Return list of floats of percentage of special characters."""
    return [len([c for c in text if c in string.punctuation]) / len(text) for text in raw_text]


def max_word_length(raw_text: Sequence[str]) -> List[int]:
    """Return list of integers of max word length."""
    result = []
    for text in raw_text:
        words = text.split()
        if not words:
            result.append(np.nan)
        result.append(max(len(w) for w in words))
    return result


def language(
    raw_text: Sequence[str],
    models_storage: Union[pathlib.Path, str, None] = None,
    lang_certainty_threshold: float = 0.8
) -> List[str]:
    """Return list of strings of language."""
    fasttext = _import_optional_property_dependency(module='fasttext', property_name='language')

    model_name = FASTTEXT_LANG_MODEL.rsplit('/', maxsplit=1)[-1]
    model_path = get_creat_model_storage(models_storage)
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
        model = fasttext.load_model(str(model_path))
    except Exception as exp:
        raise exp

    # Predictions are the first prediction (k=1), only if the probability is above the threshold
    predictions = [
        model.predict(it.replace('\n', ' '), k=1, threshold=lang_certainty_threshold)
        if it is not None
        else (None, None)
        for it in raw_text
    ]
    # labels is empty for detection below threshold
    language_codes = [
        labels[0].replace('__label__', '') if labels else None
        for labels, _ in predictions
    ]

    return language_codes


def sentiment(raw_text: Sequence[str]) -> List[str]:
    """Return list of floats of sentiment."""
    return [textblob.TextBlob(text).sentiment.polarity for text in raw_text]


def subjectivity(raw_text: Sequence[str]) -> List[str]:
    """Return list of floats of subjectivity."""
    return [textblob.TextBlob(text).sentiment.subjectivity for text in raw_text]


def _predict(text, classifier, kind):
    try:
        v = classifier(text)
    except Exception:  # pylint: disable=broad-except
        return np.nan
    else:
        if not v:
            return np.nan
        v = v[0]
        if kind == 'toxicity':
            return v['score']
        elif kind == 'fluency':
            label_value = 'LABEL_1'
        elif kind == 'formality':
            label_value = 'formal'
        else:
            raise ValueError('Unssuported value for "kind" parameter')
        return (
            v['score']
            if v['label'] == label_value
            else 1 - v['score']
        )


def toxicity(
    raw_text: Sequence[str],
    device: Optional[int] = None,
    models_storage: Union[pathlib.Path, str, None] = None
) -> List[float]:
    """Return list of floats of toxicity."""
    model_name = 'unitary/toxic-bert'
    classifier = get_transformer_pipeline(
        'toxicity',
        model_name,
        device=device,
        models_storage=models_storage
    )
    return [
        _predict(text, classifier, 'toxicity')
        for text in raw_text
    ]


def fluency(
    raw_text: Sequence[str],
    device: Optional[int] = None,
    models_storage: Union[pathlib.Path, str, None] = None
) -> List[float]:
    """Return list of floats of fluency."""
    model_name = 'prithivida/parrot_fluency_model'
    classifier = get_transformer_pipeline(
        'fluency',
        model_name,
        device=device,
        models_storage=models_storage
    )
    return [
        _predict(text, classifier, 'fluency')
        for text in raw_text
    ]


def formality(
    raw_text: Sequence[str],
    device: Optional[int] = None,
    models_storage: Union[pathlib.Path, str, None] = None
) -> List[float]:
    """Return list of floats of formality."""
    model_name = 's-nlp/roberta-base-formality-ranker'
    classifier = get_transformer_pipeline(
        'formality',
        model_name,
        device=device,
        models_storage=models_storage
    )
    return [
        _predict(text, classifier, 'formality')
        for text in raw_text
    ]


def lexical_density(raw_text: Sequence[str]) -> List[str]:
    """Return a list of floats of lexical density per text sample.

    Lexical density is the percentage of unique words in a given text. For more
    information: https://en.wikipedia.org/wiki/Lexical_density
    """
    if not nltk_download('punkt', quiet=True):
        warnings.warn('nltk punkt not found, lexical density cannot be calculated.'
                      ' Please check your internet connection.', UserWarning)
        return [np.nan] * len(raw_text)
    result = []
    for text in raw_text:
        if not pd.isna(text):
            all_words = textblob.TextBlob(text).words
            if len(all_words) == 0:
                result.append(np.nan)
            else:
                total_unique_words = len(set(all_words))
                text_lexical_density = round(total_unique_words * 100 / len(all_words), 2)
                result.append(text_lexical_density)
        else:
            result.append(np.nan)
    return result


def unique_noun_count(raw_text: Sequence[str]) -> List[float]:
    """Return a list of integers of number of unique noun words in the text."""
    if not nltk_download('averaged_perceptron_tagger', quiet=True):
        warnings.warn('nltk averaged_perceptron_tagger not found, unique noun count cannot be calculated.'
                      ' Please check your internet connection.', UserWarning)
        return [np.nan] * len(raw_text)
    result = []
    for text in raw_text:
        if not pd.isna(text):
            unique_words_with_tags = set(textblob.TextBlob(text).tags)
            result.append(sum(1 for (_, tag) in unique_words_with_tags if tag.startswith('N')))
        else:
            result.append(np.nan)
    return result


def readability_score(raw_text: Sequence[str]) -> List[float]:
    """Return a list of floats of Flesch Reading-Ease score per text sample.

    In the Flesch reading-ease test, higher scores indicate material that is easier to read
    whereas lower numbers mark texts that are more difficult to read. For more information:
    https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests#Flesch_reading_ease
    """
    if not nltk_download('punkt', quiet=True):
        warnings.warn('nltk punkt not found, readability score cannot be calculated.'
                      ' Please check your internet connection.', UserWarning)
        return [np.nan] * len(raw_text)
    if not nltk_download('cmudict', quiet=True):
        warnings.warn('nltk cmudict not found, readability score cannot be calculated.'
                      ' Please check your internet connection.', UserWarning)
        return [np.nan] * len(raw_text)
    result = []
    cmudict_dict = corpus.cmudict.dict()
    for text in raw_text:
        if not pd.isna(text):
            sentence_count = len(sent_tokenize(text))
            text = remove_punctuation(text.lower())
            words = word_tokenize(text)
            word_count = len(words)
            syllable_count = sum([len(cmudict_dict[word]) for word in words if word in cmudict_dict])
            if word_count != 0 and sentence_count != 0 and syllable_count != 0:
                avg_syllables_per_word = syllable_count / word_count
                avg_words_per_sentence = word_count / sentence_count
                flesch_reading_ease = 206.835 - (1.015 * avg_words_per_sentence) - (84.6 * avg_syllables_per_word)
                result.append(round(flesch_reading_ease, 3))
            else:
                result.append(np.nan)
        else:
            result.append(np.nan)
    return result


def average_sentence_length(raw_text: Sequence[str]) -> List[float]:
    """Return a list of floats denoting the average sentence length per text sample."""
    if not nltk_download('punkt', quiet=True):
        warnings.warn('nltk punkt not found, average sentence length cannot be calculated.'
                      ' Please check your internet connection.', UserWarning)
        return [np.nan] * len(raw_text)
    result = []
    for text in raw_text:
        if not pd.isna(text):
            sentences = [remove_punctuation(sent) for sent in sent_tokenize(text)]
            total_words = sum([len(word_tokenize(sentence)) for sentence in sentences])
            if len(sentences) != 0:
                asl = total_words / len(sentences)
                result.append(round(asl, 0))
            else:
                result.append(np.nan)
        else:
            result.append(np.nan)
    return result


def count_unique_urls(raw_text: Sequence[str]) -> List[str]:
    """Return a list of integers denoting the number of unique URLS per text sample."""
    url_pattern = r'https?:\/\/(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
    return [len(set(re.findall(url_pattern, text))) if not pd.isna(text) else 0 for text in raw_text]


def count_urls(raw_text: Sequence[str]) -> List[str]:
    """Return a list of integers denoting the number of URLS per text sample."""
    url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
    return [len(re.findall(url_pattern, text)) if not pd.isna(text) else 0 for text in raw_text]


def count_unique_email_addresses(raw_text: Sequence[str]) -> List[str]:
    """Return a list of integers denoting the number of unique email addresses per text sample."""
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
    return [len(set(re.findall(email_pattern, text))) if not pd.isna(text) else 0 for text in raw_text]


def count_email_addresses(raw_text: Sequence[str]) -> List[str]:
    """Return a list of integers denoting the number of email addresses per text sample."""
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
    return [len(re.findall(email_pattern, text)) if not pd.isna(text) else 0 for text in raw_text]


def count_unique_syllables(raw_text: Sequence[str]) -> List[str]:
    """Return a list of integers denoting the number of unique syllables per text sample."""
    if not nltk_download('punkt', quiet=True):
        warnings.warn('nltk punkt not found, readability score cannot be calculated.'
                      ' Please check your internet connection.', UserWarning)
        return [np.nan] * len(raw_text)
    if not nltk_download('cmudict', quiet=True):
        warnings.warn('nltk cmudict not found, readability score cannot be calculated.'
                      ' Please check your internet connection.', UserWarning)
        return [np.nan] * len(raw_text)
    result = []
    cmudict_dict = corpus.cmudict.dict()
    for text in raw_text:
        if not pd.isna(text):
            text = remove_punctuation(text.lower())
            words = word_tokenize(text)
            syllables = {word: True for word in words if word in cmudict_dict}
            result.append(len(syllables))
        else:
            result.append(np.nan)
    return result


def reading_time(raw_text: Sequence[str]) -> List[str]:
    """Return a list of integers denoting time in seconds to read each text sample.

    The formula is based on Demberg & Keller, 2008 where it is assumed that
    reading a character taken 14.69 milliseconds on average.
    """
    ms_per_char = 14.69
    result = []
    for text in raw_text:
        if not pd.isna(text):
            words = text.split()
            nchars = map(len, words)
            rt_per_word = map(lambda nchar: nchar * ms_per_char, nchars)
            ms_reading_time = sum(list(rt_per_word))
            result.append(round(ms_reading_time/1000, 2))
        else:
            result.append(0.00)
    return result


def sentence_length(raw_text: Sequence[str]) -> List[str]:
    """Return a list of integers denoting the number of sentences per text sample."""
    if not nltk_download('punkt', quiet=True):
        warnings.warn('nltk punkt not found, average syllable length cannot be calculated.'
                      ' Please check your internet connection.', UserWarning)
        return [np.nan] * len(raw_text)
    result = []
    for text in raw_text:
        if not pd.isna(text):
            sentence_count = len(sent_tokenize(text))
            result.append(sentence_count)
        else:
            result.append(np.nan)
    return result


def average_syllable_length(raw_text: Sequence[str]) -> List[str]:
    """Return a list of integers denoting the average number of syllables per sentences per text sample."""
    if not nltk_download('punkt', quiet=True):
        warnings.warn('nltk punkt not found, average syllable length cannot be calculated.'
                      ' Please check your internet connection.', UserWarning)
        return [np.nan] * len(raw_text)
    if not nltk_download('cmudict', quiet=True):
        warnings.warn('nltk cmudict not found, average syllable length cannot be calculated.'
                      ' Please check your internet connection.', UserWarning)
        return [np.nan] * len(raw_text)
    cmudict_dict = corpus.cmudict.dict()
    result = []
    for text in raw_text:
        if not pd.isna(text):
            sentence_count = len(sent_tokenize(text))
            text = remove_punctuation(text.lower())
            words = word_tokenize(text)
            syllable_count = sum([len(cmudict_dict[word]) for word in words if word in cmudict_dict])
            result.append(round(syllable_count/sentence_count, 2))
        else:
            result.append(np.nan)
    return result


class TextProperty(TypedDict):
    name: str
    method: Callable[..., Sequence[Any]]
    output_type: str


DEFAULT_PROPERTIES: Tuple[TextProperty, ...] = (
    {'name': 'Text Length', 'method': text_length, 'output_type': 'numeric'},
    {'name': 'Average Word Length', 'method': average_word_length, 'output_type': 'numeric'},
    {'name': 'Max Word Length', 'method': max_word_length, 'output_type': 'numeric'},
    {'name': '% Special Characters', 'method': percentage_special_characters, 'output_type': 'numeric'},
    {'name': 'Language', 'method': language, 'output_type': 'categorical'},
    {'name': 'Sentiment', 'method': sentiment, 'output_type': 'numeric'},
    {'name': 'Subjectivity', 'method': subjectivity, 'output_type': 'numeric'},
    {'name': 'Toxicity', 'method': toxicity, 'output_type': 'numeric'},
    {'name': 'Fluency', 'method': fluency, 'output_type': 'numeric'},
    {'name': 'Formality', 'method': formality, 'output_type': 'numeric'},
    {'name': 'Lexical Density', 'method': lexical_density, 'output_type': 'numeric'},
    {'name': 'Unique Noun Count', 'method': unique_noun_count, 'output_type': 'numeric'},
    {'name': 'Readability Score', 'method': readability_score, 'output_type': 'numeric'},
    {'name': 'Average Sentence Length', 'method': average_sentence_length, 'output_type': 'numeric'},
)


ALL_PROPERTIES: Tuple[TextProperty, ...] = (
    {'name': 'Count URLs', 'method': count_urls, 'output_type': 'numeric'},
    {'name': 'Count Email Address', 'method': count_email_addresses, 'output_type': 'numeric'},
    {'name': 'Count Unique URLs', 'method': count_unique_urls, 'output_type': 'numeric'},
    {'name': 'Count Unique Email Address', 'method': count_unique_email_addresses, 'output_type': 'numeric'},
    {'name': 'Count Unique Syllables', 'method': count_unique_syllables, 'output_type': 'numeric'},
    {'name': 'Reading Time', 'method': reading_time, 'output_type': 'numeric'},
    {'name': 'Sentence Length', 'method': sentence_length, 'output_type': 'numeric'},
    {'name': 'Average Syllable Length', 'method': average_syllable_length, 'output_type': 'numeric'},
) + DEFAULT_PROPERTIES


LONG_RUN_PROPERTIES = ('Toxicity', 'Fluency', 'Formality', 'Unique Noun Count')
LARGE_SAMPLE_SIZE = 10_000

ENGLISH_ONLY_PROPERTIES = (
    'Sentiment', 'Subjectivity', 'Toxicity', 'Fluency', 'Formality', 'Readability Score',
    'Unique Noun Count', 'Count Unique Syllables', 'Sentence Length', 'Average Syllable Length'
)


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
        properties = [prop for prop in all_properties if prop['name'] in include_properties]
    elif ignore_properties is not None:
        properties = [prop for prop in default_properties if prop['name'] not in ignore_properties]
    else:
        properties = default_properties

    if not include_long_calculation_properties:
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
    device: Optional[str] = None,
    models_storage: Union[pathlib.Path, str, None] = None
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
        'Readability Score', 'Average Sentence Length', 'Count URLs', Count Unique URLs', 'Count Email Address',
        'Count Unique Email Address', 'Count Unique Syllables', 'Reading Time', 'Sentence Length',
        'Average Syllable Length']
        List of default properties are: ['Text Length', 'Average Word Length', 'Max Word Length',
        '% Special Characters', 'Language', 'Sentiment', 'Subjectivity', 'Toxicity', 'Fluency', 'Formality',
        'Lexical Density', 'Unique Noun Count', 'Readability Score', 'Average Sentence Length']
        To calculate all the default properties, the include_properties and ignore_properties parameters should
        be None. If you pass either include_properties or ignore_properties then the only the properties specified
        in the list will be calculated or ignored.
        Note that the properties ['Toxicity', 'Fluency', 'Formality', 'Language', 'Unique Noun Count'] may
        take a long time to calculate. If include_long_calculation_properties is False, these properties will be
        ignored, even if they are in the include_properties parameter.
    ignore_properties : List[str], default None
        The properties to ignore from the list of default properties. If None, no properties will be ignored and
        all the default properties will be calculated. Cannot be used together with include_properties parameter.
    include_long_calculation_properties : bool, default False
        Whether to include properties that may take a long time to calculate. If False, these properties will be
        ignored, even if they are in the include_properties parameter.
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
    text_properties_names = {it['name'] for it in text_properties}
    samples_language = None
    english_samples = []
    english_samples_mask = []
    calculated_properties = {}

    if english_properties_names & text_properties_names:
        samples_language = run_available_kwargs(
            language,
            raw_text=raw_text,
            **kwargs
        )

        for lang, text in zip(samples_language, raw_text):
            if lang == 'en':
                english_samples.append(text)
                english_samples_mask.append(True)
            else:
                english_samples_mask.append(False)

        new_text_properties = []

        for prop in text_properties:
            if prop['name'] == 'Language':
                calculated_properties['Language'] = samples_language
            else:
                new_text_properties.append(prop)

        text_properties = new_text_properties

    warning_message = (
        'Failed to calculate property {0}. '
        'Dependencies required by property are not installed. '
        'Error:\n{1}'
    )

    progress_bar = create_progress_bar(
        iterable=list(text_properties),
        name='Text Properties Calculation',
        unit='Text Property'
    )

    # TODO: refactor
    for prop in progress_bar:
        progress_bar.set_postfix(
            {'Property': prop['name']},
            refresh=False
        )
        if prop['name'] not in english_properties_names:
            try:
                values = run_available_kwargs(prop['method'], raw_text=raw_text, **kwargs)
            except ImportError as e:
                warnings.warn(warning_message.format(prop['name'], str(e)))
                continue
            else:
                calculated_properties[prop['name']] = values
        else:
            try:
                values = run_available_kwargs(prop['method'], raw_text=english_samples, **kwargs)
            except ImportError as e:
                warnings.warn(warning_message.format(prop['name'], str(e)))
                continue
            else:
                result = []
                idx = 0
                fill_value = np.nan if prop['output_type'] == 'numeric' else None
                for mask in english_samples_mask:
                    if mask:
                        result.append(values[idx])
                        idx += 1
                    else:
                        result.append(fill_value)
                calculated_properties[prop['name']] = result

    if not calculated_properties:
        raise RuntimeError('Failed to calculate any of the properties.')

    properties_types = {
        k: v
        for k, v in properties_types.items()
        if k in calculated_properties
    }

    return calculated_properties, properties_types
