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
"""Module containing the text properties models for the NLP module."""
import logging
import pathlib
import warnings
from contextlib import contextmanager
from functools import lru_cache
from importlib import import_module
from typing import Optional, Union

import requests
from nltk import corpus
from transformers.utils import logging as transformers_logging

MODELS_STORAGE = pathlib.Path(__file__).absolute().parent / '.nlp-models'


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


def import_optional_property_dependency(
        module: str,
        property_name: str,
        package_name: Optional[str] = None,
        error_template: Optional[str] = None
):
    """Import additional modules in runtime."""
    try:
        lib = import_module(module)
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


def get_transformer_pipeline(
        property_name: str,
        model_name: str,
        device: Optional[str] = None,
        models_storage: Union[pathlib.Path, str, None] = None,
        use_onnx_model: bool = False,
        use_cache=False
):
    """Return a transformers' pipeline for the given model name."""
    if use_onnx_model and 'onnx' not in model_name.lower():
        raise ValueError("use_onnx_model=True, but model_name is not for a 'onnx' model")

    if use_cache:
        model, tokenizer = _get_transformer_model_and_tokenizer(property_name, model_name,
                                                                models_storage, use_onnx_model)
    else:
        # __wrapped__ is simply the function without decoration, in our case - without caching
        model, tokenizer = _get_transformer_model_and_tokenizer.__wrapped__(property_name, model_name,
                                                                            models_storage, use_onnx_model)

    if use_onnx_model:
        onnx_pipe = import_optional_property_dependency('optimum.pipelines', property_name=property_name)
        return onnx_pipe.pipeline('text-classification', model=model, tokenizer=tokenizer,
                                  accelerator='ort', device=device)
    else:
        transformers = import_optional_property_dependency('transformers', property_name=property_name)
        return transformers.pipeline('text-classification', model=model, tokenizer=tokenizer, device=device)


@contextmanager
def _log_suppressor():
    user_transformer_log_level = transformers_logging.get_verbosity()
    user_logger_level = logging.getLogger('transformers').getEffectiveLevel()
    is_progress_bar_enabled = transformers_logging.is_progress_bar_enabled()

    transformers_logging.set_verbosity_error()
    transformers_logging.disable_progress_bar()
    logging.getLogger('transformers').setLevel(50)

    with warnings.catch_warnings():
        yield

    transformers_logging.set_verbosity(user_transformer_log_level)
    logging.getLogger('transformers').setLevel(user_logger_level)
    if is_progress_bar_enabled:
        transformers_logging.enable_progress_bar()


@lru_cache(maxsize=5)
def _get_transformer_model_and_tokenizer(
        property_name: str,
        model_name: str,
        models_storage: Union[pathlib.Path, str, None] = None,
        use_onnx_model: bool = True,
):
    """Return a transformers' model and tokenizer in cpu memory."""
    transformers = import_optional_property_dependency('transformers', property_name=property_name)

    with _log_suppressor():
        models_storage = get_create_model_storage(models_storage=models_storage)
        model_path = models_storage / model_name
        model_path_exists = model_path.exists()

        if use_onnx_model:
            onnx_runtime = import_optional_property_dependency('optimum.onnxruntime', property_name=property_name)
            classifier_cls = onnx_runtime.ORTModelForSequenceClassification
            if model_path_exists:
                model = classifier_cls.from_pretrained(model_path, provider='CUDAExecutionProvider')
            else:
                model = classifier_cls.from_pretrained(model_name, provider='CUDAExecutionProvider')
                model.save_pretrained(model_path)
        else:
            if model_path_exists:
                model = transformers.AutoModelForSequenceClassification.from_pretrained(model_path)
            else:
                model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name)
                model.save_pretrained(model_path)
            model.eval()

        if model_path_exists:
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
        else:
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
            tokenizer.save_pretrained(model_path)

        return model, tokenizer


def get_cmudict_dict(use_cache=False):
    """Return corpus as dict."""
    if use_cache:
        return _get_cmudict_dict()
    return _get_cmudict_dict.__wrapped__()


@lru_cache(maxsize=1)
def _get_cmudict_dict():
    cmudict_dict = corpus.cmudict.dict()
    return cmudict_dict


FASTTEXT_LANG_MODEL = 'https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin'


def get_fasttext_model(models_storage: Union[pathlib.Path, str, None] = None, use_cache=False):
    """Return fasttext model."""
    if use_cache:
        return _get_fasttext_model(models_storage)
    return _get_fasttext_model.__wrapped__(models_storage)


@lru_cache(maxsize=1)
def _get_fasttext_model(models_storage: Union[pathlib.Path, str, None] = None):
    """Return fasttext model."""
    fasttext = import_optional_property_dependency(module='fasttext', property_name='language')

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
