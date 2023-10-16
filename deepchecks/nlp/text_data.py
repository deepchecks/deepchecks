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
"""The dataset module containing the tabular Dataset class and its functions."""
import contextlib
import pathlib
import typing as t
import warnings
from numbers import Number

import numpy as np
import pandas as pd

from deepchecks.core.errors import DeepchecksNotSupportedError, DeepchecksValueError
from deepchecks.nlp.input_validations import (ColumnTypes, validate_length_and_calculate_column_types,
                                              validate_length_and_type_numpy_array, validate_modify_label,
                                              validate_raw_text, validate_tokenized_text)
from deepchecks.nlp.task_type import TaskType, TTextLabel
from deepchecks.nlp.utils.text import break_to_lines_and_trim
from deepchecks.nlp.utils.text_data_plot import text_data_describe_plot
from deepchecks.nlp.utils.text_embeddings import calculate_builtin_embeddings
from deepchecks.nlp.utils.text_properties import calculate_builtin_properties, get_builtin_properties_types
from deepchecks.utils.logger import get_logger
from deepchecks.utils.metrics import is_label_none
from deepchecks.utils.validation import is_sequence_not_str

__all__ = ['TextData']

TDataset = t.TypeVar('TDataset', bound='TextData')


class TextData:
    """
    TextData wraps together the raw text data and the labels for the nlp task.

    The TextData class contains metadata and methods intended for easily accessing
    metadata relevant for the training or validating of ML models.

    Parameters
    ----------
    raw_text : t.Sequence[str], default: None
        The raw text data, a sequence of strings representing the raw text of each sample.
        If not given, tokenized_text must be given, and raw_text will be created from it by joining the tokens with
        spaces.
    tokenized_text : t.Sequence[t.Sequence[str]], default: None
        The tokenized text data, a sequence of sequences of strings representing the tokenized text of each sample.
        Only relevant for task_type 'token_classification'.
        If not given, raw_text must be given, and tokenized_text will be created from it by splitting the text by
        spaces.
    label : t.Optional[TTextLabel], default: None
        The label for the text data. Can be either a text_classification label or a token_classification label.
        If None, the label is not set.

        - text_classification label - For text classification the accepted label format differs between multilabel and
          single label cases. For single label data, the label should be passed as a sequence of labels, with one entry
          per sample that can be either a string or an integer. For multilabel data, the label should be passed as a
          sequence of sequences, with the sequence for each sample being a binary vector, representing the presence of
          the i-th label in that sample.
        - token_classification label - For token classification the accepted label format is the IOB format or similar
          to it. The Label must be a sequence of sequences of strings or integers, with each sequence corresponding to
          a sample in the tokenized text, and exactly the length of the corresponding tokenized text.
    task_type : str, default: None
        The task type for the text data. Can be either 'text_classification' or 'token_classification'. Must be set if
        label is provided.
    name : t.Optional[str] , default: None
        The name of the dataset. If None, the dataset name will be defined when running it within a check.
    metadata : t.Optional[t.Union[pd.DataFrame, str]] , default: None
        Metadata for the samples. Metadata must be given as a pandas DataFrame or a path to a pandas
        DataFrame compatible csv file, with the rows representing each sample
        and columns representing the different metadata columns. If None, no metadata is set.
        The number of rows in the metadata DataFrame must be equal to the number of samples in the dataset, and the
        order of the rows must be the same as the order of the samples in the dataset.
        For more on metadata, see the `NLP Metadata Guide
        <https://docs.deepchecks.com/stable/nlp/usage_guides/nlp_metadata.html>`_.
    categorical_metadata : t.Optional[t.List[str]] , default: None
        The names of the categorical metadata columns. If None, categorical metadata columns are automatically inferred.
        Only relevant if metadata is not None.
    properties : t.Optional[t.Union[pd.DataFrame, str]] , default: None
        The text properties for the samples. Properties must be given as either a pandas DataFrame or a path to a pandas
        DataFrame compatible csv file, with the rows representing each sample and columns representing the different
        properties. If None, no properties are set.
        The number of rows in the properties DataFrame must be equal to the number of samples in the dataset, and the
        order of the rows must be the same as the order of the samples in the dataset.
        In order to calculate the default properties, use the `TextData.calculate_builtin_properties` function after
        the creation of the TextData object.
        For more on properties, see the `NLP Properties Guide
        <https://docs.deepchecks.com/stable/nlp/usage_guides/nlp_properties.html>`_.
    categorical_properties : t.Optional[t.List[str]] , default: None
        The names of the categorical properties columns. Should be given only for custom properties, not for
        any of the built-in properties. If None, categorical properties columns are automatically inferred for custom
        properties.
    embeddings : t.Optional[Union[np.ndarray, pd.DataFrame, str]], default: None
        The text embeddings for the samples. Embeddings must be given as a numpy array (or a path to an .npy
        file containing a numpy array) of shape (N, E), where N is the number of samples in the TextData object and E
        is the number of embeddings dimensions.
        The numpy array must be in the same order as the samples in the TextData.
        If None, no embeddings are set.

        In order to use the built-in embeddings, use the `TextData.calculate_builtin_embeddings` function after
        the creation of the TextData object.
        For more on embeddings, see the :ref:`Text Embeddings Guide <nlp__embeddings_guide>`
    """

    _text: np.ndarray
    _label: TTextLabel
    task_type: t.Optional[TaskType]
    _tokenized_text: t.Optional[t.Sequence[t.Sequence[str]]] = None  # Outer sequence is np array
    name: t.Optional[str] = None
    _embeddings: t.Optional[t.Union[pd.DataFrame, str]] = None
    _metadata: t.Optional[t.Union[pd.DataFrame, str]] = None
    _properties: t.Optional[t.Union[pd.DataFrame, str]] = None
    _cat_properties: t.Optional[t.List[str]] = None
    _cat_metadata: t.Optional[t.List[str]] = None
    _numeric_metadata: t.Optional[t.List[str]] = None
    _original_text_index: t.Optional[t.Sequence[int]] = None  # Sequence is np array

    def __init__(
            self,
            raw_text: t.Optional[t.Sequence[str]] = None,
            tokenized_text: t.Optional[t.Sequence[t.Sequence[str]]] = None,
            label: t.Optional[TTextLabel] = None,
            task_type: t.Optional[str] = None,
            name: t.Optional[str] = None,
            embeddings: t.Optional[t.Union[pd.DataFrame, np.ndarray, str]] = None,
            metadata: t.Optional[pd.DataFrame] = None,
            categorical_metadata: t.Optional[t.List[str]] = None,
            properties: t.Optional[pd.DataFrame] = None,
            categorical_properties: t.Optional[t.List[str]] = None,
    ):
        # Require explicitly setting task type if label is provided
        if task_type in [None, 'other']:
            if label is not None:
                raise DeepchecksValueError('task_type must be set when label is provided')
            self._task_type = TaskType.OTHER
        elif task_type == 'text_classification':
            self._task_type = TaskType.TEXT_CLASSIFICATION
        elif task_type == 'token_classification':
            if tokenized_text is None:
                raise DeepchecksValueError('tokenized_text must be provided for token_classification task type')
            validate_tokenized_text(tokenized_text)
            modified = [[str(token) for token in tokens_per_sample] for tokens_per_sample in tokenized_text]
            self._tokenized_text = np.asarray(modified, dtype=object)
            self._task_type = TaskType.TOKEN_CLASSIFICATION
        else:
            raise DeepchecksNotSupportedError(f'task_type {task_type} is not supported, must be one of '
                                              'text_classification, token_classification, other')

        if raw_text is None:
            if tokenized_text is None:
                raise DeepchecksValueError('Either raw_text or tokenized_text must be provided')
            self._text = np.asarray([' '.join(tokens) for tokens in tokenized_text])  # Revisit this decision
        else:
            validate_raw_text(raw_text)
            self._text = np.asarray([str(x) for x in raw_text])
            if tokenized_text is not None and len(raw_text) != len(tokenized_text):
                raise DeepchecksValueError('raw_text and tokenized_text sequences must have the same length')

        self._label = validate_modify_label(label, self._task_type, len(self), tokenized_text)

        if name is not None and not isinstance(name, str):
            raise DeepchecksNotSupportedError(f'name must be a string, got {type(name)}')
        self.name = name

        if metadata is not None:
            self.set_metadata(metadata, categorical_metadata)
        if properties is not None:
            self.set_properties(properties, categorical_properties)
        if embeddings is not None:
            self.set_embeddings(embeddings)

        # Used for display purposes
        self._original_text_index = np.arange(len(self))

    def is_multi_label_classification(self) -> bool:
        """Check if the dataset is multi-label."""
        if self.task_type == TaskType.TEXT_CLASSIFICATION and self._label is not None:
            return is_sequence_not_str(self._label[0])
        return False

    # pylint: disable=protected-access
    def copy(self: TDataset, rows_to_use: t.Optional[t.Sequence[int]] = None) -> TDataset:
        """Create a copy of this Dataset with new data.

        Parameters
        ----------
        rows_to_use : t.Optional[t.List[int]] , default: None
            The rows to use in the new copy. If None, the new copy will contain all the rows.
        """
        cls = type(self)

        # NOTE:
        # Make sure we won't get the warning for setting class in the non multilabel case
        with disable_deepchecks_logger():

            if rows_to_use is None:
                new_copy = cls(
                    raw_text=self._text,
                    tokenized_text=self._tokenized_text,
                    label=self._label,
                    task_type=self._task_type.value,
                    name=self.name
                )

                if self._metadata is not None:
                    new_copy.set_metadata(self._metadata, self._cat_metadata)

                if self._properties is not None:
                    new_copy.set_properties(self._properties, self._cat_properties)

                if self._embeddings is not None:
                    new_copy.set_embeddings(self._embeddings)

                new_copy._original_text_index = self._original_text_index
                return new_copy

            if not isinstance(rows_to_use, t.Sequence) or any(not isinstance(x, Number) for x in rows_to_use):
                raise DeepchecksValueError('rows_to_use must be a list of integers')

            rows_to_use = sorted(rows_to_use)

            new_copy = cls(
                raw_text=self._text[rows_to_use],
                tokenized_text=(
                    self._tokenized_text[rows_to_use]
                    if self._tokenized_text is not None
                    else None
                ),
                label=self._label[rows_to_use] if self.has_label() else None,
                task_type=self._task_type.value, name=self.name
            )

            if self._metadata is not None:
                metadata = self._metadata.iloc[rows_to_use, :]
                new_copy.set_metadata(metadata, self._cat_metadata)

            if self._properties is not None:
                properties = self._properties.iloc[rows_to_use, :]
                new_copy.set_properties(properties, self._cat_properties)

            if self._embeddings is not None:
                embeddings = self._embeddings[rows_to_use]
                new_copy.set_embeddings(embeddings)

            new_copy._original_text_index = self._original_text_index[rows_to_use]
            return new_copy

    def sample(self: TDataset, n_samples: int, replace: bool = False, random_state: t.Optional[int] = None,
               drop_na_label: bool = False) -> TDataset:
        """Create a copy of the dataset object, with the internal data being a sample of the original data.

        Parameters
        ----------
        n_samples : int
            Number of samples to draw.
        replace : bool, default: False
            Whether to sample with replacement.
        random_state : t.Optional[int] , default None
            Random state.
        drop_na_label : bool, default: False
            Whether to take sample only from rows with exiting label.

        Returns
        -------
        Dataset
            instance of the Dataset with sampled internal dataframe.
        """
        samples_to_choose_from = np.arange(len(self))
        if drop_na_label and self.has_label():
            samples_to_choose_from = samples_to_choose_from[[not is_label_none(x) for x in self._label]]
        n_samples = min(n_samples, len(samples_to_choose_from))

        np.random.seed(random_state)
        sample_idx = np.random.choice(samples_to_choose_from, n_samples, replace=replace)
        return self.copy(rows_to_use=sorted(sample_idx))

    def __len__(self) -> int:
        """Return number of samples in the dataset."""
        return self.n_samples

    @property
    def n_samples(self) -> int:
        """Return number of samples in the dataset."""
        if self._text is not None:
            return len(self._text)
        elif self._label is not None:
            return len(self._label)
        else:
            return 0

    @property
    def embeddings(self) -> pd.DataFrame:
        """Return the embeddings of for the dataset."""
        if self._embeddings is None:
            raise DeepchecksValueError(
                'Functionality requires embeddings, but the the TextData object had none. To use this functionality, '
                'use the set_embeddings method to set your own embeddings with a numpy.array or use '
                'TextData.calculate_builtin_embeddings to add the default deepchecks embeddings.'
            )
        return self._embeddings

    def calculate_builtin_embeddings(self, model: str = 'miniLM', file_path: str = 'embeddings.npy',
                                     device: t.Optional[str] = None, long_sample_behaviour: str = 'average+warn',
                                     open_ai_batch_size: int = 500):
        """Calculate the built-in embeddings of the dataset.

        Parameters
        ----------
        model : str, default: 'miniLM'
            The model to use for calculating the embeddings. Possible values are:
            'miniLM': using the miniLM model in the sentence-transformers library.
            'open_ai': using the ADA model in the open_ai library. Requires an API key.
        file_path : str, default: 'embeddings.npy'
            The path to save the embeddings to.
        device : str, default: None
            The device to use for calculating the embeddings. If None, the default device will be used.
        long_sample_behaviour : str, default 'average+warn'
            How to handle long samples. Averaging is done as described in
            https://github.com/openai/openai-cookbook/blob/main/examples/Embedding_long_inputs.ipynb
            Currently, applies only to the 'open_ai' model, as the 'miniLM' model can handle long samples.

            Options are:
                - 'average+warn' (default): average the embeddings of the chunks and warn if the sample is too long.
                - 'average': average the embeddings of the chunks.
                - 'truncate': truncate the sample to the maximum length.
                - 'raise': raise an error if the sample is too long.
                - 'nan': return an embedding vector of nans for each sample that is too long.
        open_ai_batch_size : int, default 500
            The amount of samples to send to open ai in each batch. Reduce if getting errors from open ai.
        """
        if self._embeddings is not None:
            warnings.warn('Embeddings already exist, overwriting them', UserWarning)

        self._embeddings = calculate_builtin_embeddings(text=self.text, model=model, file_path=file_path, device=device,
                                                        long_sample_behaviour=long_sample_behaviour,
                                                        open_ai_batch_size=open_ai_batch_size)

    def set_embeddings(self, embeddings: np.ndarray, verbose: bool = True):
        """Set the embeddings of the dataset.

        Parameters
        ----------
        embeddings : pd.DataFrame
            Embeddings to set.
        verbose : bool, default: True
            Whether to print information about the process.
        """
        if self._embeddings is not None and verbose is True:
            warnings.warn('Embeddings already exist, overwriting it', UserWarning)

        if isinstance(embeddings, pd.DataFrame):
            embeddings = embeddings.to_numpy()

        if isinstance(embeddings, str):
            embeddings = np.load(embeddings)

        if embeddings is not None:
            validate_length_and_type_numpy_array(embeddings, 'Embeddings', len(self))
        self._embeddings = embeddings

    @property
    def metadata(self) -> pd.DataFrame:
        """Return the metadata of for the dataset."""
        if self._metadata is None:
            raise DeepchecksValueError(
                'Functionality requires metadata, but the the TextData object had none. '
                'To use this functionality, use the '
                'set_metadata method to set your own metadata with a pandas.DataFrame.'
            )
        return self._metadata

    @property
    def categorical_metadata(self) -> t.List[str]:
        """Return categorical metadata column names."""
        return self._cat_metadata

    @property
    def numerical_metadata(self) -> t.List[str]:
        """Return numeric metadata column names."""
        return self._numeric_metadata

    def set_metadata(
            self,
            metadata: pd.DataFrame,
            categorical_metadata: t.Optional[t.Sequence[str]] = None
    ):
        """Set the metadata of the dataset."""
        if self._metadata is not None:
            warnings.warn('Metadata already exist, overwriting it', UserWarning)

        if isinstance(metadata, str):
            metadata = pd.read_csv(metadata)

        column_types = validate_length_and_calculate_column_types(
            data_table=metadata,
            data_table_name='Metadata',
            expected_size=len(self),
            categorical_columns=categorical_metadata
        )

        self._metadata = metadata.reset_index(drop=True)
        self._cat_metadata = column_types.categorical_columns
        self._numeric_metadata = column_types.numerical_columns

    def calculate_builtin_properties(
            self,
            include_properties: t.Optional[t.List[str]] = None,
            ignore_properties: t.Optional[t.List[str]] = None,
            include_long_calculation_properties: bool = False,
            ignore_non_english_samples_for_english_properties: bool = True,
            device: t.Optional[str] = None,
            models_storage: t.Union[pathlib.Path, str, None] = None,
            batch_size: t.Optional[int] = 16,
            cache_models: bool = False,
            use_onnx_models: bool = True,
    ):
        """Calculate the default properties of the dataset.

        Parameters
        ----------
        include_properties : List[str], default None
            The properties to calculate. If None, all default properties will be calculated. Cannot be used
            together with ignore_properties parameter. Available properties are:
            ['Text Length', 'Average Word Length', 'Max Word Length',
            '% Special Characters', '% Punctuation', 'Language',
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
            If True, will use onnx gpu optimized models for the calculation. Requires the optimum[onnxruntime-gpu]
            library to be installed as well as the availability of GPU.
        """
        if self._properties is not None:
            warnings.warn('Properties already exist, overwriting them', UserWarning)

        properties, properties_types = calculate_builtin_properties(
            list(self.text),
            include_properties=include_properties,
            ignore_properties=ignore_properties,
            include_long_calculation_properties=include_long_calculation_properties,
            ignore_non_english_samples_for_english_properties=ignore_non_english_samples_for_english_properties,
            device=device,
            models_storage=models_storage,
            batch_size=batch_size,
            cache_models=cache_models,
            use_onnx_models=use_onnx_models,
        )

        self._properties = pd.DataFrame(properties, index=self.get_original_text_indexes())
        self._cat_properties = [k for k, v in properties_types.items() if v == 'categorical']

    def set_properties(
            self,
            properties: pd.DataFrame,
            categorical_properties: t.Optional[t.Sequence[str]] = None
    ):
        """Set the properties of the dataset."""
        if self._properties is not None:
            warnings.warn('Properties already exist, overwriting them', UserWarning)

        if categorical_properties is not None:
            categories_not_in_data = set(categorical_properties).difference(properties.columns.tolist())
            if not len(categories_not_in_data) == 0:
                raise DeepchecksValueError(
                    f'The following columns does not exist in Properties - {list(categories_not_in_data)}'
                )

        if isinstance(properties, str):
            properties = pd.read_csv(properties)

        builtin_property_types = get_builtin_properties_types()
        property_names = properties.columns.tolist()
        intersection = set(builtin_property_types.keys()).intersection(property_names)

        # Get column types for intersection properties
        builtin_categorical_properties = [x for x in intersection if builtin_property_types[x] == 'categorical']

        # Get column types for user properties
        user_properties = list(set(property_names).difference(builtin_property_types.keys()))
        if categorical_properties is None:
            user_categorical_properties = None
        else:
            user_categorical_properties = list(set(categorical_properties).intersection(user_properties))

        if len(user_properties) != 0:
            column_types = validate_length_and_calculate_column_types(
                data_table=properties[user_properties],
                data_table_name='Properties',
                expected_size=len(self),
                categorical_columns=user_categorical_properties
            )
        else:
            column_types = ColumnTypes([], [])

        # merge the two categorical properties list into one ColumnTypes object
        all_cat_properties = column_types.categorical_columns + builtin_categorical_properties
        column_types = ColumnTypes(
            categorical_columns=all_cat_properties,
            numerical_columns=list(set(property_names).difference(all_cat_properties))
        )

        self._properties = properties.reset_index(drop=True)
        self._cat_properties = column_types.categorical_columns

    def save_properties(self, path: str):
        """Save the dataset properties to csv.

        Parameters
        ----------
        path : str
            Path to save the properties to.
        """
        if self._properties is None:
            raise DeepchecksNotSupportedError(
                'TextData does not contain properties, add them by using '
                '"calculate_builtin_properties" or "set_properties" functions'
            )

        self._properties.to_csv(path, index=False)

    @property
    def properties(self) -> pd.DataFrame:
        """Return the properties of the dataset."""
        if self._properties is None:
            raise DeepchecksNotSupportedError(
                'Functionality requires properties, but the the TextData object had none. To use this functionality, '
                'use the set_properties method to set your own properties with a pandas.DataFrame or use '
                'TextData.calculate_builtin_properties to add the default deepchecks properties.'
            )
        return self._properties

    @property
    def categorical_properties(self) -> t.List[str]:
        """Return categorical properties names."""
        return self._cat_properties

    @property
    def numerical_properties(self) -> t.List[str]:
        """Return numerical properties names."""
        if self._properties is not None:
            return [prop for prop in self._properties.columns if prop not in self._cat_properties]
        else:
            return []

    @property
    def task_type(self) -> t.Optional[TaskType]:
        """Return the task type.

        Returns
        -------
        t.Optional[TaskType]
            Task type
        """
        return self._task_type

    @property
    def text(self) -> t.Sequence[str]:
        """Return sequence of raw text samples.

        Returns
        -------
        t.Sequence[str]
           Sequence of raw text samples.
        """
        return self._text

    @property
    def tokenized_text(self) -> t.Sequence[t.Sequence[str]]:
        """Return sequence of tokenized text samples.

        Returns
        -------
        t.Sequence[t.Sequence[str]]
           Sequence of tokenized text samples.
        """
        if self._tokenized_text is None:
            raise DeepchecksValueError('Tokenized text is not set, provide it when initializing the TextData object '
                                       'to run the requested functionalities')
        return self._tokenized_text

    @property
    def label(self) -> TTextLabel:
        """Return the label defined in the dataset.

        Returns
        -------
        TTextLabel
        """
        if not self.has_label():
            raise DeepchecksValueError('Label is not set, provide it when initializing the TextData object '
                                       'to run the requested functionalities')
        return self._label

    def label_for_display(self, model_classes: list = None) -> TTextLabel:
        """Return the label defined in the dataset in a format that can be displayed.

        Parameters
        ----------
        model_classes : list, default None
            List of classes names to use for multi-label display. Only used if the dataset is multi-label.

        Returns
        -------
        TTextLabel
        """
        if self.is_multi_label_classification():
            ret_labels = [np.argwhere(x == 1).flatten().tolist() for x in self.label]
            if model_classes:
                ret_labels = [[model_classes[i] for i in x] for x in ret_labels]
            return ret_labels
        else:
            return self.label

    def label_for_print(self, model_classes: list = None) -> t.List[str]:
        """Return the label defined in the dataset in a format that can be printed nicely.

        Parameters
        ----------
        model_classes : list, default None
            List of classes names to use for multi-label display. Only used if the dataset is multi-label.

        Returns
        -------
        List[str]
        """
        label_for_display = self.label_for_display(model_classes)
        return [break_to_lines_and_trim(str(x)) for x in label_for_display]

    def has_label(self) -> bool:
        """Return True if label was set.

        Returns
        -------
        bool
           True if label was set.
        """
        return self._label is not None

    def get_original_text_indexes(self) -> t.Sequence[int]:
        """Return the original indexes of the text samples.

        Returns
        -------
        t.Sequence[int]
           Original indexes of the text samples.
        """
        assert self._original_text_index is not None, 'Internal Error'
        return self._original_text_index

    def get_sample_at_original_index(self, index: int) -> str:
        """Return the text sample at the original index.

        Parameters
        ----------
        index : int
            Original index of the text sample.

        Returns
        -------
        str
           Text sample at the original index.
        """
        locations_in_array = np.where(self._original_text_index == index)
        if len(locations_in_array) == 0:
            raise DeepchecksValueError('Original text index is not in sampled TextData object')
        elif len(locations_in_array) > 1:
            raise DeepchecksValueError('Original text index is not unique in sampled TextData object')
        return self._text[int(locations_in_array[0])]

    @classmethod
    def cast_to_dataset(cls, obj: t.Any) -> 'TextData':
        """Verify Dataset or transform to Dataset.

        Function verifies that provided value is a non-empty instance of Dataset,
        otherwise raises an exception, but if the 'cast' flag is set to True it will
        also try to transform provided value to the Dataset instance.

        Parameters
        ----------
        obj
            value to verify

        Raises
        ------
        DeepchecksValueError
            if the provided value is not a TextData instance;
            if the provided value cannot be transformed into Dataset instance;
        """
        if not isinstance(obj, cls):
            raise DeepchecksValueError(f'{obj} is not a {cls.__name__} instance')
        return obj.copy()

    def validate_textdata_compatibility(self, other_text_data: 'TextData') -> bool:
        """Verify that all provided datasets share same label name and task types.

        Parameters
        ----------
        other_text_data : TextData
            The other dataset TextData object to compare with.

        Returns
        -------
        bool
            True if provided dataset share same label name and task types.
        """
        assert other_text_data is not None
        if self.task_type != other_text_data.task_type:
            return False

        return True

    def head(self, n_samples: int = 5, model_classes: list = None) -> pd.DataFrame:
        """Return a copy of the dataset as a pandas Dataframe with the first n_samples samples.

        Parameters
        ----------
        n_samples : int, default 5
            Number of samples to return.
        model_classes : list, default None
            List of classes names to use for multi-label display. Only used if the dataset is multi-label.

        Returns
        -------
        pd.DataFrame
            A copy of the dataset as a pandas Dataframe with the first n_samples samples.
        """
        if n_samples > len(self):
            n_samples = len(self) - 1
        result = pd.DataFrame({'text': self.text[:n_samples]}, index=self.get_original_text_indexes()[:n_samples])
        if self.has_label():
            result['label'] = self.label_for_display(model_classes=model_classes)[:n_samples]
        if self._tokenized_text is not None:
            result['tokenized_text'] = self.tokenized_text[:n_samples]
        if self._metadata is not None:
            result = result.join(self.metadata.loc[result.index])
        return result

    def len_when_sampled(self, n_samples: t.Optional[int]):
        """Return number of samples in the sampled dataframe this dataset is sampled with n_samples samples."""
        if n_samples is None:
            return self.n_samples
        return min(self.n_samples, n_samples)

    def is_sampled(self, n_samples: t.Optional[int]):
        """Return True if the dataset number of samples will decrease when sampled with n_samples samples."""
        if n_samples is None:
            return False
        return self.n_samples > n_samples

    def describe(self, n_properties_to_show: t.Optional[int] = 4, properties_to_show: t.Optional[t.List[str]] = None,
                 max_num_labels_to_show: t.Optional[int] = 5, model_classes: t.Optional[t.List[str]] = None):
        """Provide holistic view of the data.

        Generates the following plots:
        1. Label distribution
        2. Statistics about the data such as number of samples, annotation ratio, list of metadata columns, list of
        text properties and so on.
        3. Property distribution for the text properties defined either by n_properties_to_show or properties_to_show
        parameter.

        Parameters
        ----------
        n_properties_to_show : int, default: 4
            Number of properties to consider for generating property distribution graphs. If properties_to_show
            is provided, this value is ignored.
        properties_to_show : List[str], default: None
            List of property names to consider for generating property distribution graphs. If None, all the
            properties are considered.
        max_num_labels_to_show : int, default: 5
            The threshold to display the maximum number of labels on the label distribution pie chart and
            display rest of the labels under "Others" category.
        model_classes : Optional[List[str]], default: None
            List of classes names to use for multi-label display. Only used if the dataset is multi-label.

        Returns
        -------
        Displays the Plotly Figure.
        """
        prop_names = []
        all_properties_data = pd.DataFrame()
        if self._properties is None and properties_to_show is not None:
            raise DeepchecksValueError('No properties exist!')
        elif self._properties is not None:
            if properties_to_show is not None:
                prop_names = [prop for prop in properties_to_show if prop in self.properties.columns]
                if len(prop_names) != len(properties_to_show):
                    raise DeepchecksValueError(f'{set(properties_to_show) - set(prop_names)} '
                                               'properties does not exist in the TextData object')
            else:
                prop_names = list(self.properties.columns)[:n_properties_to_show]
            all_properties_data = self.properties[prop_names]

        fig = text_data_describe_plot(properties=all_properties_data, n_samples=self.n_samples,
                                      is_multi_label=self.is_multi_label_classification(), task_type=self.task_type,
                                      categorical_metadata=self.categorical_metadata,
                                      numerical_metadata=self.numerical_metadata,
                                      categorical_properties=self.categorical_properties,
                                      numerical_properties=self.numerical_properties, label=self._label,
                                      model_classes=model_classes,
                                      max_num_labels_to_show=max_num_labels_to_show)

        return fig


@contextlib.contextmanager
def disable_deepchecks_logger():
    """Disable deepchecks root logger."""
    logger = get_logger()
    logger_state = logger.disabled
    logger.disabled = True
    yield
    logger.disabled = logger_state
