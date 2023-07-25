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
import importlib
import pathlib
from functools import lru_cache
from typing import Optional, Union

import requests
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


def _get_transformer_model(
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
        transformers = import_optional_property_dependency('transformers', property_name=property_name)
        # TODO: quantize if 'quantize_model' is True
        return transformers.AutoModelForSequenceClassification.from_pretrained(
            model_name,
            cache_dir=models_storage,
            device_map=device
        )

    onnx = import_optional_property_dependency(
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
            return onnx.ORTModelForSequenceClassification.from_pretrained(model_path).to(device or -1)

        model = onnx.ORTModelForSequenceClassification.from_pretrained(
            model_name,
            export=True,
            cache_dir=models_storage,
        ).to(device or -1)
        # NOTE:
        # 'optimum', after exporting/converting a model to the ONNX format,
        # does not store it onto disk we need to save it now to not reconvert
        # it each time
        model.save_pretrained(model_path)
        return model

    model_path = models_storage / 'onnx' / 'quantized' / model_name

    if model_path.exists():
        return onnx.ORTModelForSequenceClassification.from_pretrained(model_path).to(device or -1)

    not_quantized_model = _get_transformer_model(
        property_name,
        model_name,
        device,
        quantize_model=False,
        models_storage=models_storage
    )

    quantizer = onnx.ORTQuantizer.from_pretrained(not_quantized_model).to(device or -1)

    quantizer.quantize(
        save_dir=model_path,
        # TODO: make it possible to provide a config as a parameter
        quantization_config=onnx.configuration.AutoQuantizationConfig.avx512_vnni(
            is_static=False,
            per_channel=False
        )
    )
    return onnx.ORTModelForSequenceClassification.from_pretrained(model_path).to(device or -1)


def import_optional_property_dependency(
        module: str,
        property_name: str,
        package_name: Optional[str] = None,
        error_template: Optional[str] = None
):
    """Import additional modules in runtime."""
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


def get_transformer_pipeline(
        property_name: str,
        model_name: str,
        device: Optional[str] = None,
        models_storage: Union[pathlib.Path, str, None] = None,
        use_cache=False
):
    """Return a transformers pipeline for the given model name."""
    if use_cache:
        return _get_transformer_pipeline(property_name, model_name, device, models_storage)
    # __wrapped__ is simply the function without decoration, in our case - without caching
    return _get_transformer_pipeline.__wrapped__(property_name, model_name, device, models_storage)


@lru_cache(maxsize=5)
def _get_transformer_pipeline(
        property_name: str,
        model_name: str,
        device: Optional[str] = None,
        models_storage: Union[pathlib.Path, str, None] = None
):
    """Return a transformers pipeline for the given model name."""
    transformers = import_optional_property_dependency('transformers', property_name=property_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, device_map=device)
    model = _get_transformer_model(
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
