# ----------------------------------------------------------------------------
# Copyright (C) 2021-2022 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""Module for base nlp context."""
import collections
import typing as t
from operator import itemgetter

import numpy as np

from deepchecks.core.context import BaseContext
from deepchecks.core.errors import (DatasetValidationError, DeepchecksNotSupportedError, DeepchecksValueError,
                                    ModelValidationError, ValidationError)
from deepchecks.nlp.metric_utils.scorers import init_validate_scorers
from deepchecks.nlp.metric_utils.token_classification import (SpanAligner, get_default_token_scorers, get_scorer_dict,
                                                              validate_scorers)
from deepchecks.nlp.task_type import TaskType
from deepchecks.nlp.text_data import TextData
from deepchecks.nlp.utils.data_inference import infer_observed_and_model_labels
from deepchecks.tabular.utils.task_type import TaskType as TabularTaskType

__all__ = [
    'Context',
    'TTextPred',
    'TTextProba',
    'TTokenPred'
]

from deepchecks.tabular.metric_utils import DeepcheckScorer, get_default_scorers
from deepchecks.tabular.utils.validation import ensure_predictions_proba, ensure_predictions_shape
from deepchecks.utils.docref import doclink
from deepchecks.utils.logger import get_logger
from deepchecks.utils.typing import BasicModel

TClassPred = t.Union[t.Sequence[t.Union[str, int]], t.Sequence[t.Sequence[t.Union[str, int]]]]
TClassProba = t.Sequence[t.Sequence[float]]
TTokenPred = t.Sequence[t.Sequence[t.Tuple[str, int, int, float]]]
TTextPred = t.Union[TClassPred, TTokenPred]
TTextProba = t.Union[TClassProba]


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

    def __init__(self,
                 test: TextData,
                 y_pred_test: TTextPred,
                 y_proba_test: TTextProba,
                 train: t.Union[TextData, None] = None,
                 y_pred_train: TTextPred = None,
                 y_proba_train: TTextProba = None,
                 model_classes: list = None,
                 validate_data_on_predict: bool = True):
        """Initialize dummy model."""
        predictions = {}
        probas = {}

        if ((y_proba_train is not None) or (y_proba_test is not None)) and \
                (train.task_type == TaskType.TOKEN_CLASSIFICATION):
            raise DeepchecksNotSupportedError('For token classification probabilities should be provided as the 4th'
                                              'element in the token annotation tuples within the predictions argument '
                                              'of the rum method, while the probabilities argument of the run method'
                                              ' should be left empty.')

        if train is not None and test is not None:
            # check if datasets have same indexes
            if set(train.index) & set(test.index):
                train.index = map(lambda x: f'train-{x}', list(train.index))
                test.index = map(lambda x: f'test-{x}', list(test.index))
                get_logger().warning('train and test datasets have common index - adding "train"/"test"'
                                     ' prefixes. To avoid that provide datasets with no common indexes '
                                     'or pass the model object instead of the predictions.')

        for dataset, y_pred, y_proba in zip([train, test],
                                            [y_pred_train, y_pred_test],
                                            [y_proba_train, y_proba_test]):
            if dataset is not None:
                if y_pred is not None:
                    self._validate_prediction(dataset, y_pred, len(model_classes))
                if y_proba is not None:
                    self._validate_proba(dataset, y_proba, len(model_classes))

                if dataset.task_type == TaskType.TEXT_CLASSIFICATION:
                    if (y_pred is None) and (y_proba is not None):
                        if dataset.is_multilabel:
                            y_pred = (np.array(y_proba) > 0.5)  # TODO: Replace with user-configurable threshold
                            y_pred = [np.array(model_classes)[pred] for pred in y_pred]
                        else:
                            y_pred = np.argmax(np.array(y_proba), axis=-1)
                            y_pred = np.array(model_classes)[y_pred]

                    if y_pred is not None:
                        y_pred = np.array(y_pred)
                        if len(y_pred.shape) > 1 and y_pred.shape[1] == 1:
                            y_pred = y_pred[:, 0]
                        ensure_predictions_shape(y_pred, dataset.text)

                    if y_proba is not None:
                        ensure_predictions_proba(y_proba, y_pred)
                        y_proba_dict = dict(zip(dataset.index, y_proba))
                        probas.update({dataset.name: y_proba_dict})

                if y_pred is not None:
                    y_pred_dict = dict(zip(dataset.index, y_pred))
                    predictions.update({dataset.name: y_pred_dict})

        self.predictions = predictions
        self.probas = probas
        self.validate_data_on_predict = validate_data_on_predict
        self._classes = model_classes

        if self.predictions:
            self.predict = self._predict
            self._prediction_indices = \
                {name: set(data_preds.keys()) for name, data_preds in self.predictions.items()}

        if self.probas:
            self.predict_proba = self._predict_proba
            self._proba_indices = \
                {name: set(data_proba.keys()) for name, data_proba in self.probas.items()}

    def _predict(self, data: TextData) -> TTextPred:  # TODO: Needs to receive list of strings, not TextData
        """Predict on given data by the data indexes."""
        if self.validate_data_on_predict:
            data_indices = set(np.random.choice(data.index, min(100, len(data.index)), replace=False))
            if not data_indices.issubset(self._prediction_indices[data.name]):
                raise DeepchecksValueError('Data that has not been seen before passed for inference with pre computed '
                                           'predictions.')
        return list(itemgetter(*data.index)(self.predictions[data.name]))  # pylint: disable=unsubscriptable-object

    def _predict_proba(self, data: TextData) -> TTextProba:  # TODO: Needs to receive list of strings, not TextData
        """Predict probabilities on given data by the data indexes."""
        if self.validate_data_on_predict:
            data_indices = set(np.random.choice(data.index, min(100, len(data.index)), replace=False))
            if not data_indices.issubset(self._proba_indices[data.name]):
                raise DeepchecksValueError('Data that has not been seen before passed for inference with pre computed '
                                           'probabilities.')
        return list(itemgetter(*data.index)(self.probas[data.name]))  # pylint: disable=unsubscriptable-object

    def fit(self, *args, **kwargs):
        """Just for python 3.6 (sklearn validates fit method)."""
        pass

    @staticmethod
    def _validate_prediction(dataset: TextData, prediction: TTextPred, n_classes: int):
        """Validate prediction for given dataset."""
        if not isinstance(prediction, collections.abc.Sequence):
            ValidationError(f'Check requires predictions for {dataset.name} to be a sequence')
        if len(prediction) != dataset.n_samples:
            raise ValidationError(f'Check requires predictions for {dataset.name} to have '
                                  f'{dataset.n_samples} rows, same as dataset')

        if dataset.task_type == TaskType.TEXT_CLASSIFICATION:
            _DummyModel._validate_classification_prediction(dataset, prediction, n_classes)
        elif dataset.task_type == TaskType.TOKEN_CLASSIFICATION:
            _DummyModel._validate_token_classification_prediction(dataset, prediction)

    @staticmethod
    def _validate_classification_prediction(dataset: TextData, prediction: TTextPred, n_classes: int):
        """Validate prediction for given text classification dataset."""
        classification_format_error = f'Check requires classification for {dataset.name} to be ' \
                                      f'either a sequence that can be cast to a 1D numpy array of shape' \
                                      f' (n_samples,), or a sequence of sequences that can be cast to a 2D ' \
                                      f'numpy array of shape (n_samples, n_classes) for the multilabel case.'

        try:
            prediction = np.array(prediction)
            if dataset.is_multilabel:
                prediction = prediction.astype(float)  # Multilabel prediction is a binary matrix
            else:
                prediction = prediction.reshape((-1, 1))  # Multiclass (not multilabel) Prediction can be a string
                if prediction.shape[0] != dataset.n_samples:
                    raise ValidationError(classification_format_error)
        except ValueError as e:
            raise ValidationError(classification_format_error) from e
        pred_shape = prediction.shape
        if dataset.is_multilabel:
            if len(pred_shape) == 1 or pred_shape[1] != n_classes:
                raise ValidationError(classification_format_error)
            if not np.array_equal(prediction, prediction.astype(bool)):
                raise ValidationError(f'Check requires classification predictions for {dataset.name} dataset '
                                      f'to be either 0 or 1')

    @staticmethod
    def _validate_token_classification_prediction(dataset: TextData, prediction: TTextPred):
        """Validate prediction for given token classification dataset."""
        if not all(isinstance(pred, collections.abc.Sequence) for pred in prediction):
            raise ValidationError(f'Check requires token classification for {dataset.name} to be a sequence '
                                  f'of sequences')
        for sample_pred in prediction:
            for token_pred in sample_pred:
                if len(token_pred) != 4:
                    raise ValidationError(f'Check requires token classification for {dataset.name} to have '
                                          f'4 entries')
                if not isinstance(token_pred[1], int) or not isinstance(token_pred[2], int):
                    raise ValidationError(f'Check requires token classification for {dataset.name} to have '
                                          f'int indices representing the start and end of the token at the second'
                                          f' and third entry in the token prediction tuples')
                if not token_pred[1] < token_pred[2]:
                    raise ValidationError(f'Check requires token classification predictions for {dataset.name} to have '
                                          f'token span start before span end')
                if not isinstance(token_pred[3], float) or (token_pred[3] < 0. or token_pred[3] > 1.):
                    raise ValidationError(f'Check requires token classification for {dataset.name} to have '
                                          f'probabilities between 0 and 1, at the fourth entry in the token '
                                          f'prediction tuples')

    @staticmethod
    def _validate_proba(dataset: TextData, probabilities: TTextProba, n_classes: int,
                        eps: float = 1e-3):
        """Validate predicted probabilities for given dataset."""
        classification_format_error = f'Check requires classification probabilities for {dataset.name} to be a ' \
                                      f'sequence of sequences that can be cast to a 2D numpy array of shape' \
                                      f' (n_samples, n_classes)'

        if len(probabilities) != dataset.n_samples:
            raise ValidationError(f'Check requires classification probabilities for {dataset.name} dataset '
                                  f'to have {dataset.n_samples} rows, same as dataset')

        if dataset.task_type == TaskType.TEXT_CLASSIFICATION:
            try:
                probabilities = np.array(probabilities, dtype='float')
            except ValueError as e:
                raise ValidationError(classification_format_error) from e
            proba_shape = probabilities.shape
            if len(proba_shape) != 2:
                raise ValidationError(classification_format_error)
            if proba_shape[1] != n_classes:
                raise ValidationError(f'Check requires classification probabilities for {dataset.name} dataset '
                                      f'to have {n_classes} columns, same as the number of classes')
            if dataset.is_multilabel:
                if (probabilities > 1).any() or (probabilities < 0).any():
                    raise ValidationError(f'Check requires classification probabilities for {dataset.name} '
                                          f'dataset to be between 0 and 1')
            else:
                if any(abs(probabilities.sum(axis=1) - 1) > eps):
                    raise ValidationError(f'Check requires classification probabilities for {dataset.name} '
                                          f'dataset to be probabilities and sum to 1 for each row')


class Context(BaseContext):
    """Contains all the data + properties the user has passed to a check/suite, and validates it seamlessly.

    Parameters
    ----------
    train_dataset : Union[TextData, None], default: None
        TextData object, representing data an estimator was fitted on
    test_dataset : Union[TextData, None], default: None
        TextData object, representing data an estimator predicts on
    with_display : bool, default: True
        flag that determines if checks will calculate display (redundant in some checks).
    train_pred : Union[TTextPred, None], default: None
        predictions on train dataset
    test_pred : Union[TTextPred, None], default: None
        predictions on test dataset
    train_proba : Union[TTextProba, None], default: None
        probabilities on train dataset
    test_proba : Union[TTextProba, None], default: None
        probabilities on test dataset
    model_classes : Optional[List], default: None
        list of classes known to the model
    random_state: int, default 42
        A seed to set for pseudo-random functions, primarily sampling.
    n_samples: int, default: 10_000
        The number of samples to use within the checks.
    """

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
            n_samples: t.Optional[int] = 10_000
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
            if test_dataset.has_label() and train_dataset.has_label() and not \
                    TextData.datasets_share_task_type(train_dataset, test_dataset):
                raise DatasetValidationError('train_dataset and test_dataset must share the same label and task type')
        if test_dataset and not train_dataset:
            raise DatasetValidationError('Can\'t initialize context with only test_dataset. if you have single '
                                         'dataset, initialize it as train_dataset')
        if model_classes and len(model_classes) == 0:
            raise DeepchecksValueError('Received empty model_classes')
        if model_classes and sorted(model_classes) != model_classes:
            supported_models_link = doclink(
                'nlp-supported-predictions-format',
                template='For more information please refer to the Supported Tasks guide {link}')
            raise DeepchecksValueError(f'Received unsorted model_classes. {supported_models_link}')

        self._task_type = train_dataset.task_type if train_dataset else test_dataset.task_type if test_dataset else None

        self._observed_classes, self._model_classes = \
            infer_observed_and_model_labels(train_dataset=train_dataset, test_dataset=test_dataset,
                                            model_classes=model_classes)

        if any(x is not None for x in (train_pred, test_pred, train_proba, test_proba)):
            self._model = _DummyModel(train=train_dataset, test=test_dataset,
                                      y_pred_train=train_pred, y_pred_test=test_pred,
                                      y_proba_train=train_proba, y_proba_test=test_proba,
                                      model_classes=self.model_classes)
        else:
            self._model = None

        self._train = train_dataset
        self._test = test_dataset
        if n_samples is not None and self._train is not None:
            self._train = self._train.sample(n_samples, random_state=random_state)
        if n_samples is not None and self._test is not None:
            self._test = self._test.sample(n_samples, random_state=random_state)

        self._validated_model = False
        self._with_display = with_display

        # Init a span aligner object for the run
        self._span_aligner = SpanAligner()

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
            # If in infer_task_type we didn't find classes on model, or user didn't pass any, then using the observed
            self._model_classes = self._observed_classes
            get_logger().warning('Could not find model\'s classes, using the observed classes')
        return self._model_classes

    @property
    def observed_classes(self) -> t.List:
        """Return the observed classes in both train and test."""
        return self._observed_classes

    @property
    def span_aligner(self) -> SpanAligner:
        """Return the cached SpanAligner object."""
        return self._span_aligner

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

    def get_scorers(self,
                    scorers: t.Union[t.Mapping[str, t.Union[str, t.Callable]], t.List[str]] = None,
                    use_avg_defaults=True,
                    span_aligner: t.Optional[SpanAligner] = None) -> t.List[DeepcheckScorer]:
        """Return initialized & validated scorers in a given priority.

        If receive `scorers` use them,
        Else if user defined global scorers use them,
        Else use default scorers.

        Parameters
        ----------
        scorers : Union[List[str], Dict[str, Union[str, Callable]]], default: None
            List of scorers to use. If None, use default scorers.
            Scorers can be supplied as a list of scorer names or as a dictionary of names and functions.
        use_avg_defaults : bool, default: True
            If no scorers were provided, for classification, determines whether to use default scorers that return
            an averaged metric, or default scorers that return a metric per class.
        span_aligner: t.Optional[SpanAligner], default: None
            A SpanAligner object for processing the raw token classification annotations to the seqeval accepted format
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
            scoring_dict = get_scorer_dict(span_aligner)
            if scorers is None:
                scorers = get_default_token_scorers(use_avg_defaults)  # Get string names of default scorers
            else:
                validate_scorers(scorers, span_aligner)  # Validate that use supplied scorer names are OK
            scorers = {name: scoring_dict[name] for name in scorers}
        else:
            raise DeepchecksValueError(f'Task type must be either {TaskType.TEXT_CLASSIFICATION} or '
                                       f'{TaskType.TOKEN_CLASSIFICATION} but received {self.task_type}')
        return init_validate_scorers(scorers, self.model_classes, self._observed_classes)
