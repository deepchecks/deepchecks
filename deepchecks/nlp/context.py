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
"""Module for base nlp context."""
import typing as t
from operator import itemgetter

import numpy as np

from deepchecks.core.context import BaseContext
from deepchecks.core.errors import (DatasetValidationError, DeepchecksNotSupportedError, DeepchecksValueError,
                                    ModelValidationError)
from deepchecks.nlp.input_validations import (_validate_multilabel, _validate_text_classification,
                                              _validate_token_classification, compare_dataframes)
from deepchecks.nlp.metric_utils.scorers import init_validate_scorers
from deepchecks.nlp.metric_utils.token_classification import (get_default_token_scorers, get_scorer_dict,
                                                              validate_scorers)
from deepchecks.nlp.task_type import TaskType
from deepchecks.nlp.text_data import TextData
from deepchecks.nlp.utils.data_inference import infer_observed_and_model_labels
from deepchecks.tabular.metric_utils import DeepcheckScorer, get_default_scorers
from deepchecks.tabular.utils.task_type import TaskType as TabularTaskType
from deepchecks.utils.docref import doclink
from deepchecks.utils.logger import get_logger
from deepchecks.utils.typing import BasicModel
from deepchecks.utils.validation import is_sequence_not_str

__all__ = [
    'Context',
    'TTextPred',
    'TTextProba',
    'TTokenPred'
]


TClassPred = t.Union[
    t.Sequence[int],
    t.Sequence[str],
    t.Sequence[t.Sequence[int]]
]
TTokenPred = t.Union[
    t.Sequence[t.Sequence[int]],
    t.Sequence[t.Sequence[str]],
]

TTextPred = t.Union[TClassPred, TTokenPred]
TTextProba = t.Sequence[t.Sequence[float]]


class _DummyModel(BasicModel):
    """Dummy model class used for inference with static predictions from the user.

    Parameters
    ----------
    train: TextData
        Dataset, representing data an estimator was fitted on.
    test: TextData
        Dataset, representing data an estimator predicts on.
    y_pred_train: np.ndarray
        Array of the model prediction over the train dataset.
    y_pred_test: np.ndarray
        Array of the model prediction over the test dataset.
    y_proba_train: np.ndarray
        Array of the model prediction probabilities over the train dataset.
    y_proba_test: np.ndarray
        Array of the model prediction probabilities over the test dataset.
    validate_data_on_predict: bool, default = True
        If true, before predicting validates that the received data samples have the same index as in original data.
    """

    predictions: t.Dict[str, t.Dict[int, TTextPred]]
    proba: t.Dict[str, t.Dict[int, TTextProba]]

    def __init__(
        self,
        *,
        test: TextData,
        y_pred_test: TTextPred,
        y_proba_test: TTextProba,
        model_classes: t.List[t.Any],
        train: t.Optional[TextData] = None,
        y_pred_train: t.Optional[TTextPred] = None,
        y_proba_train: t.Optional[TTextProba] = None,
        validate_data_on_predict: bool = True,
        multilabel_proba_threshold: float = 0.5
    ):
        """Initialize dummy model."""
        predictions = {}
        probas = {}

        if train is not None and test is not None:
            # check if datasets have same indexes
            train_index = train.get_original_text_indexes()
            test_index = test.get_original_text_indexes()
            if set(train_index) & set(test_index):
                train._original_text_index = np.asarray([f'train-{i}' for i in train_index])
                test._original_text_index = np.asarray([f'test-{i}' for i in test_index])
                # # This is commented out as currently text data indices are len(range(len(data)))
                # # TODO: Uncomment when text data indices are not len(range(len(data)))
                # get_logger().warning('train and test datasets have common index - adding "train"/"test"'
                #                      ' prefixes. To avoid that provide datasets with no common indexes '
                #                      'or pass the model object instead of the predictions.')

        for dataset, y_pred, y_proba in (
            (train, y_pred_train, y_proba_train),
            (test, y_pred_test, y_proba_test),
        ):
            if dataset is None:
                continue

            if dataset.is_multi_label_classification():
                y_pred, y_proba = _validate_multilabel(
                    dataset=dataset,
                    predictions=y_pred,
                    probabilities=y_proba,
                    n_of_classes=len(model_classes)
                )
                if y_pred is None and y_proba is not None:
                    y_pred = (np.array(y_proba) > multilabel_proba_threshold)
                    y_pred = y_pred.astype(int)

            elif dataset.task_type is TaskType.TEXT_CLASSIFICATION:
                y_pred, y_proba = _validate_text_classification(
                    dataset=dataset,
                    predictions=y_pred,
                    probabilities=y_proba,
                    n_of_classes=len(model_classes)
                )
                if y_pred is None and y_proba is not None:
                    y_pred = np.argmax(np.array(y_proba), axis=-1)
                    y_pred = np.array(model_classes, dtype='str')[y_pred]

            elif dataset.task_type is TaskType.TOKEN_CLASSIFICATION:
                _validate_token_classification(
                    dataset=dataset,
                    predictions=y_pred,
                    probabilities=y_proba,
                )

            else:
                raise ValueError(f'Unknown task type - {type(dataset.task_type)}')

            if y_pred is not None:
                y_pred_dict = dict(zip(dataset.get_original_text_indexes(), y_pred))
                predictions.update({dataset.name: y_pred_dict})
            if y_proba is not None:
                y_proba_dict = dict(zip(dataset.get_original_text_indexes(), y_proba))
                probas.update({dataset.name: y_proba_dict})

        self.predictions = predictions
        self.probas = probas
        self.validate_data_on_predict = validate_data_on_predict
        self._classes = model_classes

        if self.predictions:
            self.predict = self._predict
            self._prediction_indices = {
                name: set(data_preds.keys())
                for name, data_preds in self.predictions.items()
            }
        if self.probas:
            self.predict_proba = self._predict_proba
            self._proba_indices = {
                name: set(data_proba.keys())
                for name, data_proba in self.probas.items()
            }

    def _predict(self, data: TextData) -> TTextPred:  # TODO: Needs to receive list of strings, not TextData
        """Predict on given data by the data indexes."""
        if self.validate_data_on_predict:
            data_indices = set(np.random.choice(data.get_original_text_indexes(), min(100, len(data)), replace=False))
            if not data_indices.issubset(self._prediction_indices[data.name]):
                raise DeepchecksValueError('Data that has not been seen before passed for inference with pre computed '
                                           'predictions.')
        return list(itemgetter(*data.get_original_text_indexes())(
            self.predictions[data.name]))  # pylint: disable=unsubscriptable-object

    def _predict_proba(self, data: TextData) -> TTextProba:  # TODO: Needs to receive list of strings, not TextData
        """Predict probabilities on given data by the data indexes."""
        if self.validate_data_on_predict:
            data_indices = set(np.random.choice(data.get_original_text_indexes(), min(100, len(data)), replace=False))
            if not data_indices.issubset(self._proba_indices[data.name]):
                raise DeepchecksValueError('Data that has not been seen before passed for inference with pre computed '
                                           'probabilities.')
        return list(itemgetter(*data.get_original_text_indexes())(
            self.probas[data.name]))  # pylint: disable=unsubscriptable-object

    def fit(self, *args, **kwargs):
        """Just for python 3.6 (sklearn validates fit method)."""
        pass


class Context(BaseContext):
    """Contains all the data + properties the user has passed to a check/suite, and validates it seamlessly.

    Parameters
    ----------
    train_dataset : Union[TextData, None] , default: None
        TextData object, representing data an estimator was fitted on
    test_dataset : Union[TextData, None] , default: None
        TextData object, representing data an estimator predicts on
    with_display : bool , default: True
        flag that determines if checks will calculate display (redundant in some checks).
    train_pred : Union[TTextPred, None] , default: None
        predictions on train dataset
    test_pred : Union[TTextPred, None] , default: None
        predictions on test dataset
    train_proba : Union[TTextProba, None] , default: None
        probabilities on train dataset
    test_proba : Union[TTextProba, None] , default: None
        probabilities on test dataset
    model_classes : Optional[List] , default: None
        list of classes known to the model
    random_state: int, default 42
        A seed to set for pseudo-random functions , primarily sampling.
    """

    _common_datasets_properties: t.Optional[t.Dict[str, str]] = None

    def __init__(
            self,
            train_dataset: t.Union[TextData, None] = None,
            test_dataset: t.Union[TextData, None] = None,
            with_display: bool = True,
            train_pred: t.Optional[TTextPred] = None,
            test_pred: t.Optional[TTextPred] = None,
            train_proba: t.Optional[TTextProba] = None,
            test_proba: t.Optional[TTextProba] = None,
            model_classes: t.Optional[t.List] = None,
            random_state: int = 42,
    ):
        # Validations
        if train_dataset is None and test_dataset is None:
            raise DatasetValidationError('Check must be given at least one dataset')
        if train_dataset is not None:
            train_dataset = TextData.cast_to_dataset(train_dataset)
            if train_dataset.name is None:
                train_dataset.name = 'Train'
        if test_dataset is not None:
            test_dataset = TextData.cast_to_dataset(test_dataset)
            if test_dataset.name is None:
                test_dataset.name = 'Test'

        # If both dataset, validate they fit each other
        if train_dataset and test_dataset:
            if (
                test_dataset.has_label()
                and train_dataset.has_label()
                and not train_dataset.validate_textdata_compatibility(test_dataset)
            ):
                raise DatasetValidationError('train_dataset and test_dataset must share the same label and task type')

            if (
                train_dataset._properties is not None
                and test_dataset._properties is not None
            ):
                result = compare_dataframes(
                    train=train_dataset._properties,
                    test=test_dataset._properties,
                    train_categorical_columns=train_dataset._cat_properties,
                    test_categorical_columns=test_dataset._cat_properties
                )
                self._common_datasets_properties = result.common
                if result.difference is not None:
                    get_logger().warning(
                        'Train and Test datasets have different properties, '
                        'only common properties will be used in train-test checks.\n'
                        'Properties present only in train dataset: %s\n'
                        'Properties present only in test dataset: %s\n'
                        'Properties with different types: %s\n',
                        result.difference.only_in_train,
                        result.difference.only_in_test,
                        result.difference.types_mismatch
                    )

        if test_dataset and not train_dataset:
            raise DatasetValidationError('Can\'t initialize context with only test_dataset. if you have single '
                                         'dataset, initialize it as train_dataset')

        if model_classes is not None:
            if (not is_sequence_not_str(model_classes)) or len(model_classes) == 0:
                raise DeepchecksValueError('model_classes must be a non-empty sequence')
            if sorted(model_classes) != model_classes:
                supported_models_link = doclink(
                    'nlp-supported-predictions-format',
                    template='For more information please refer to the Supported Tasks guide {link}')
                raise DeepchecksValueError(f'Received unsorted model_classes. {supported_models_link}')

        self._task_type = self.infer_task_type(train_dataset, test_dataset)

        self._observed_classes, self._model_classes = \
            infer_observed_and_model_labels(train_dataset=train_dataset, test_dataset=test_dataset,
                                            model=None, y_pred_train=train_pred, y_pred_test=test_pred,
                                            model_classes=model_classes, task_type=self.task_type)

        if any(x is not None for x in (train_pred, test_pred, train_proba, test_proba)):
            self._model = _DummyModel(train=train_dataset, test=test_dataset,
                                      y_pred_train=train_pred, y_pred_test=test_pred,
                                      y_proba_train=train_proba, y_proba_test=test_proba,
                                      model_classes=self.model_classes)
        else:
            self._model = None

        self._train = train_dataset
        self._test = test_dataset
        self._validated_model = False
        self._with_display = with_display
        self.random_state = random_state

    @property
    def common_datasets_properties(self) -> t.Dict[str, str]:
        """Return common for train and test datasets properties."""
        if self._common_datasets_properties is None:
            raise DeepchecksNotSupportedError(
                'Check is irrelevant without providing train and test '
                'datasets with precalculated properties'
            )
        return dict(self._common_datasets_properties)

    @property
    def model(self) -> _DummyModel:
        """Return model if exists, otherwise raise error."""
        if self._model is None:
            raise DeepchecksNotSupportedError('Check is irrelevant without providing predictions')
        return self._model

    @property
    def model_name(self) -> str:
        """Return the name of the model."""
        return 'Pre-computed predictions'

    @property
    def model_classes(self) -> t.List:
        """Return ordered list of possible label classes for classification tasks."""
        if self._model_classes is None:
            # If in infer_observed_and_model_labels we didn't find classes on model, or user didn't pass any,
            # then using the observed
            self._model_classes = self._observed_classes
            get_logger().warning('Could not find model\'s classes, using the observed classes. '
                                 'In order to make sure the classes used by the model are inferred correctly, '
                                 'please use the model_classes argument')
        return self._model_classes

    @property
    def observed_classes(self) -> t.List:
        """Return the observed classes in both train and test."""
        return self._observed_classes

    @staticmethod
    def infer_task_type(train_dataset: TextData, test_dataset: TextData):
        """Infer the task type."""
        if not test_dataset:
            return train_dataset.task_type
        elif train_dataset.task_type != test_dataset.task_type:
            raise DeepchecksValueError(f'datasets must have the same task type. Received '
                                       f'{train_dataset.task_type.value} for train and '
                                       f'{test_dataset.task_type.value} for test')
        return train_dataset.task_type

    @property
    def task_type(self) -> TaskType:
        """Return task type if model & train_dataset & label exists. otherwise, raise error."""
        if self._task_type is None:
            if self._train is not None and self._test is not None:
                if self._train.task_type != self._test.task_type:
                    raise DatasetValidationError('train_dataset and test_dataset have different task types')
            self._task_type = self.train.task_type
        return self._task_type

    def have_test(self):
        """Return whether there is test_dataset dataset defined."""
        return self._test is not None

    def assert_task_type(self, *expected_types: TaskType):
        """Assert task_type matching given types.

        If task_type is defined, validate it and raise error if needed, else returns True.
        If task_type is not defined, return False.
        """
        if self.task_type not in expected_types:
            raise ModelValidationError(
                f'Check is relevant for models of type {[e.value.lower() for e in expected_types]}, '
                f"but received model of type '{self.task_type.value.lower()}'"  # pylint: disable=inconsistent-quotes
            )
        return True

    def raise_if_token_classification_task(self, check=None):
        """Raise an exception if it is a token classification task."""
        check_name = type(check).__name__ if check else 'Check'
        task_type_name = TaskType.TOKEN_CLASSIFICATION.value
        if self.task_type is TaskType.TOKEN_CLASSIFICATION:
            raise DeepchecksNotSupportedError(
                f'"{check_name}" is not supported for the "{task_type_name}" tasks'
            )

    def is_multi_label_task(self):
        """Return whether the task is multi-label classification."""
        if self.task_type == TaskType.TEXT_CLASSIFICATION:
            dataset = t.cast(TextData, self._train if self._train is not None else self._test)
            return dataset.is_multi_label_classification()
        return False

    def raise_if_multi_label_task(self, check=None):
        """Raise an exception if it is a multi-label classification task."""
        check_name = type(check).__name__ if check else 'Check'
        if self.is_multi_label_task():
            raise DeepchecksNotSupportedError(
                f'"{check_name}" is not supported for the multilable classification tasks'
            )

    def get_scorers(self,
                    scorers: t.Union[t.Mapping[str, t.Union[str, t.Callable]], t.List[str], None] = None,
                    use_avg_defaults=True) -> t.List[DeepcheckScorer]:
        """Return initialized & validated scorers if provided or default scorers otherwise.

        Parameters
        ----------
        scorers : Union[List[str], Dict[str, Union[str, Callable]]], default: None
            List of scorers to use. If None, use default scorers.
            Scorers can be supplied as a list of scorer names or as a dictionary of names and functions.
        use_avg_defaults : bool, default: True
            If no scorers were provided, for classification, determines whether to use default scorers that return
            an averaged metric, or default scorers that return a metric per class.
        Returns
        -------
        List[DeepcheckScorer]
            A list of initialized & validated scorers.
        """
        if self.task_type == TaskType.TEXT_CLASSIFICATION:
            if len(self.model_classes) > 2:
                scorers = scorers or get_default_scorers(TabularTaskType.MULTICLASS, use_avg_defaults)
            else:
                scorers = scorers or get_default_scorers(TabularTaskType.BINARY, use_avg_defaults)
        elif self.task_type == TaskType.TOKEN_CLASSIFICATION:
            if scorers is None:
                scorers = get_default_token_scorers(use_avg_defaults)  # Get string names of default scorers
            else:
                validate_scorers(scorers)  # Validate that use supplied scorer names are OK
            scoring_dict = get_scorer_dict()
            scorers = {name: scoring_dict[name] for name in scorers}
        else:
            raise DeepchecksValueError(f'Task type must be either {TaskType.TEXT_CLASSIFICATION} or '
                                       f'{TaskType.TOKEN_CLASSIFICATION} but received {self.task_type}')
        return init_validate_scorers(scorers, self.model_classes, self._observed_classes)

    def get_single_scorer(self,
                          scorer: t.Union[t.Mapping[str, t.Union[str, t.Callable]], t.List[str], None] = None,
                          use_avg_defaults=True) -> DeepcheckScorer:
        """Return initialized & validated scorer if provided or a default scorer otherwise.

        Parameters
        ----------
        scorer : t.Union[t.Mapping[str, t.Union[str, t.Callable]], t.List[str], None], default: None
            List of scorers to use. If None, use default scorers.
            Scorers can be supplied as a list of scorer names or as a dictionary of names and functions.
        use_avg_defaults : bool, default: True
            If no scorers were provided, for classification, determines whether to use default scorers that return
            an averaged metric, or default scorers that return a metric per class.
        Returns
        -------
        List[DeepcheckScorer]
            An initialized & validated scorer.
        """
        return self.get_scorers(scorer, use_avg_defaults)[0]
