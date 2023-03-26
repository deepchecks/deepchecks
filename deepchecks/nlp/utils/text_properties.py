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
import string
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from deepchecks.utils.function import run_available_kwargs

__all__ = ['calculate_default_properties']


def property_import_error(property_name: str, package_name: str) -> ImportError:
    """Raise an ImportError for a property that requires a package."""
    return ImportError(
        f'property {property_name} requires the {package_name} python package. '
        f'To get it, run "pip install {package_name}". You may install the dependencies for all text properties '
        f'by running "pip install deepchecks[nlp-properties]"')


def get_transformer_model(property_name: str, model_name: str, device: Optional[str] = None):
    """Get the transformer model and decide if to use optimum.onnxruntime.

    optimum.onnxruntime is used to optimize running times on CPU.
    """
    if device is None or device == 'cpu':
        try:
            from optimum.onnxruntime import ORTModelForSequenceClassification  # pylint: disable=import-outside-toplevel
        except ImportError as e:
            error_message = f'The device was set to {device} while computing the {property_name} property, in which' \
                            f' case deepchecks resorts to accelerating the inference using optimum. Either set the ' \
                            f'device according to your hardware, or install the dependencies for all text properties ' \
                            f'by running "pip install deepchecks[nlp-properties]"'
            raise error_message from e

        return ORTModelForSequenceClassification.from_pretrained(model_name, export=True)
    else:
        try:
            from transformers import AutoModelForSequenceClassification  # pylint: disable=import-outside-toplevel
        except ImportError as e:
            raise property_import_error(property_name, 'transformers') from e
        return AutoModelForSequenceClassification.from_pretrained(model_name)


def get_transformer_pipeline(property_name: str, model_name: str, device: Optional[str] = None):
    """Return a transformers pipeline for the given model name."""
    try:
        from transformers import AutoTokenizer, pipeline  # pylint: disable=import-outside-toplevel
    except ImportError as e:
        raise property_import_error(property_name, 'transformers') from e
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = get_transformer_model(property_name, model_name, device)
    return pipeline('text-classification', model=model, tokenizer=tokenizer, device=device)


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
    return [max([len(word) for word in text.split()]) for text in raw_text]


def language(raw_text: Sequence[str]) -> List[str]:
    """Return list of strings of language."""
    try:
        import langdetect  # pylint: disable=import-outside-toplevel
        from langdetect import DetectorFactory  # pylint: disable=import-outside-toplevel
        DetectorFactory.seed = 42
    except ImportError as e:
        raise property_import_error('language', 'langdetect') from e

    return [langdetect.detect(text) for text in raw_text]


def sentiment(raw_text: Sequence[str]) -> List[str]:
    """Return list of floats of sentiment."""
    try:
        import textblob  # pylint: disable=import-outside-toplevel
    except ImportError as e:
        raise property_import_error('sentiment', 'textblob') from e

    return [textblob.TextBlob(text).sentiment.polarity for text in raw_text]


def subjectivity(raw_text: Sequence[str]) -> List[str]:
    """Return list of floats of subjectivity."""
    try:
        import textblob  # pylint: disable=import-outside-toplevel
    except ImportError as e:
        raise property_import_error('subjectivity', 'textblob') from e

    return [textblob.TextBlob(text).sentiment.subjectivity for text in raw_text]


def toxicity(raw_text: Sequence[str], device: Optional[int] = None) -> List[float]:
    """Return list of floats of toxicity."""
    model_name = 'unitary/toxic-bert'
    classifier = get_transformer_pipeline('toxicity', model_name, device=device)
    return [x['score'] for x in classifier(raw_text)]


def fluency(raw_text: Sequence[str], device: Optional[int] = None) -> List[float]:
    """Return list of floats of fluency."""
    model_name = 'prithivida/parrot_fluency_model'
    classifier = get_transformer_pipeline('fluency', model_name, device=device)
    return [x['score'] if x['label'] == 'LABEL_1' else 1 - x['score'] for x in classifier(raw_text)]


def formality(raw_text: Sequence[str], device: Optional[int] = None) -> List[float]:
    """Return list of floats of formality."""
    model_name = 's-nlp/roberta-base-formality-ranker'
    classifier = get_transformer_pipeline('formality', model_name, device=device)
    return [x['score'] if x['label'] == 'formal' else 1 - x['score'] for x in classifier(raw_text)]


DEFAULT_PROPERTIES = [
    {'name': 'Text Length', 'method': text_length, 'output_type': 'numeric'},
    {'name': 'Average Word Length', 'method': average_word_length, 'output_type': 'numeric'},
    {'name': 'Max Word Length', 'method': max_word_length, 'output_type': 'numeric'},
    {'name': '% Special Characters', 'method': percentage_special_characters, 'output_type': 'numeric'},
    {'name': 'Language', 'method': language, 'output_type': 'categorical'},
    {'name': 'Sentiment', 'method': sentiment, 'output_type': 'numeric'},
    {'name': 'Subjectivity', 'method': subjectivity, 'output_type': 'numeric'},
    {'name': 'Toxicity', 'method': toxicity, 'output_type': 'numeric'},
    {'name': 'Fluency', 'method': fluency, 'output_type': 'numeric'},
    {'name': 'Formality', 'method': formality, 'output_type': 'numeric'}
]


LONG_RUN_PROPERTIES = ['Toxicity', 'Fluency', 'Formality']
ENGLISH_ONLY_PROPERTIES = ['Sentiment', 'Subjectivity', 'Toxicity', 'Fluency', 'Formality']
LARGE_SAMPLE_SIZE = 10_000


def _get_default_properties(include_properties: List[str] = None, ignore_properties: List[str] = None):
    """Return the default properties.

    Default properties are defined here and not outside the function so not to import all the packages
    if they are not needed.
    """
    ret_properties = DEFAULT_PROPERTIES

    # Filter by properties or ignore_properties:
    if include_properties is not None and ignore_properties is not None:
        raise ValueError('Cannot use properties and ignore_properties parameters together.')
    elif include_properties is not None:
        ret_properties = [prop for prop in ret_properties if prop['name'] in include_properties]
    elif ignore_properties is not None:
        ret_properties = [prop for prop in ret_properties if prop['name'] not in ignore_properties]

    return ret_properties


def calculate_default_properties(raw_text: Sequence[str], include_properties: Optional[List[str]] = None,
                                 ignore_properties: Optional[List[str]] = None, device: Optional[str] = None
                                 ) -> Tuple[Dict[str, List[float]], Dict[str, str]]:
    """Return list of dictionaries of text properties.

    Parameters
    ----------
        raw_text : Sequence[str]
            The text to calculate the properties for.
        include_properties : List[str], default None
            The properties to calculate. If None, all default properties will be calculated. Cannot be used together
            with ignore_properties parameter.
        ignore_properties : List[str], default None
            The properties to ignore. If None, no properties will be ignored. Cannot be used together with
            properties parameter.
        device : str, default None
            The device to use for the calculation. If None, the default device will be used.

    Returns
    -------
        Dict[str, List[float]]
            A dictionary with the property name as key and a list of the property values for each text as value.
        Dict[str, str]
            A dictionary with the property name as key and the property's type as value.
    """
    default_text_properties = _get_default_properties(include_properties=include_properties,
                                                      ignore_properties=ignore_properties)

    # Check if the run may take a long time and warn
    heavy_properties = [prop for prop in default_text_properties if prop['name'] in LONG_RUN_PROPERTIES]
    if heavy_properties and len(raw_text) > LARGE_SAMPLE_SIZE:
        h_property_names = [prop['name'] for prop in heavy_properties]
        warning_message = f'Calculating the properties {h_property_names} on a large dataset may take a long time.' \
                          f' Consider using a smaller sample size or running this code on better hardware.'
        if device is None or device == 'cpu':
            warning_message += ' Consider using a GPU or a similar device to run these properties.'

        warnings.warn(warning_message, UserWarning)

    calculated_properties = {}
    for prop in default_text_properties:
        try:
            res = run_available_kwargs(prop['method'], raw_text=raw_text, device=device)
            calculated_properties[prop['name']] = res
        except ImportError as e:
            warnings.warn(f'Failed to calculate property {prop["name"]}. Error: {e}')
    if not calculated_properties:
        raise RuntimeError('Failed to calculate any of the properties.')

    properties_types = {prop['name']: prop['output_type'] for prop in default_text_properties}  # TODO: Add tests

    return calculated_properties, properties_types
