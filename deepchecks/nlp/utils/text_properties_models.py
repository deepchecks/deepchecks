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
import pathlib
from functools import lru_cache
from importlib import import_module
from importlib.util import find_spec
from typing import Optional, Union

import requests
import torch
from nltk import corpus

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
        quantize_model: bool = True,
        use_cache=False
):
    """Return a transformers' pipeline for the given model name."""
    transformers = import_optional_property_dependency('transformers', property_name=property_name)
    if use_cache:
        model, tokenizer = _get_transformer_model_and_tokenizer(property_name, model_name,
                                                                models_storage, quantize_model)
    else:
        # __wrapped__ is simply the function without decoration, in our case - without caching
        model, tokenizer = _get_transformer_model_and_tokenizer.__wrapped__(property_name, model_name,
                                                                            models_storage, quantize_model)

    pipeline_kwargs = {'device_map': 'auto'} if find_spec('accelerate') is not None else {'device': device}
    return transformers.pipeline('text-classification', model=model, tokenizer=tokenizer, **pipeline_kwargs)


@lru_cache(maxsize=5)
def _get_transformer_model_and_tokenizer(
        property_name: str,
        model_name: str,
        models_storage: Union[pathlib.Path, str, None] = None,
        quantize_model: bool = True,
):
    """Return a transformers' model and tokenizer in cpu memory."""
    transformers = import_optional_property_dependency('transformers', property_name=property_name)
    models_storage = get_create_model_storage(models_storage=models_storage)

    model_kwargs = dict(device_map=None)
    if quantize_model:
        model_kwargs['load_in_8bit'] = True
        model_kwargs['torch_dtype'] = torch.float32
        model_path = models_storage / 'quantized' / model_name
    else:
        model_path = models_storage / model_name

    if model_path.exists():
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, device_map=None)
        model = transformers.AutoModelForSequenceClassification.from_pretrained(model_path, **model_kwargs)
    else:
        model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name, **model_kwargs)
        model.save_pretrained(model_path)

        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, device_map=None)
        tokenizer.save_pretrained(model_path)

    model.eval()
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
