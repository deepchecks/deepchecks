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

from deepchecks.core import CheckFailure, CheckResult, DatasetKind
from deepchecks.core.errors import (DatasetValidationError, DeepchecksNotSupportedError, DeepchecksValueError,
                                    ModelValidationError, ValidationError)
from deepchecks.nlp.metric_utils.scorers import init_validate_scorers
from deepchecks.nlp.task_type import TaskType
from deepchecks.nlp.text_data import TextData, TTextLabel
from deepchecks.tabular.utils.task_type import TaskType as TabularTaskType

__all__ = [
    'Context',
    'TTextPred',
    'TTextProba'
]

from deepchecks.tabular.metric_utils import DeepcheckScorer, get_default_scorers
from deepchecks.tabular.utils.validation import ensure_predictions_proba, ensure_predictions_shape
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
    train: Dataset
        Dataset, representing data an estimator was fitted on.
    test: Dataset
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
                 validate_data_on_predict: bool = True):
        """Initialize dummy model."""
        if train is not None:
            if train.name is None:
                train.name = 'train'
        if test is not None:
            if test.name is None:
                test.name = 'test'

        predictions = {}
        probas = {}

        if ((y_proba_train is not None) or (y_proba_test is not None)) and\
                (train.task_type == TaskType.TOKEN_CLASSIFICATION):
            raise DeepchecksNotSupportedError('For token classification probabilities should be part of the token'
                                              ' prediction annotation and not passed to the proba argument')

        for dataset, y_pred, y_proba in zip([train, test],
                                            [y_pred_train, y_pred_test],
                                            [y_proba_train, y_proba_test]):
            if dataset is not None:
                if y_pred is not None:
                    self._validate_prediction(dataset, y_pred)
                if y_proba is not None:
                    self._validate_proba(dataset, y_proba)

                if dataset.task_type == TaskType.TEXT_CLASSIFICATION:
                    if (y_pred is None) and (y_proba is not None) and\
                            (dataset.task_type == TaskType.TEXT_CLASSIFICATION):
                        if dataset.is_multilabel:
                            y_pred = np.array(y_proba) > 0.5
                        else:
                            y_pred = np.argmax(np.array(y_proba), axis=-1)
                    else:
                        y_pred = np.array(y_pred)
                        if len(y_pred.shape) > 1 and y_pred.shape[1] == 1:
                            y_pred = y_pred[:, 0]
                        ensure_predictions_shape(y_pred, dataset.text)

                y_pred_dict = dict(zip(dataset.index, y_pred))
                predictions[dataset.name] = y_pred_dict
                if y_proba is not None:
                    ensure_predictions_proba(y_proba, y_pred)
                    y_proba_dict = dict(zip(dataset.index, y_proba))
                    probas[dataset.name] = y_proba_dict

        self.predictions = predictions if predictions else None
        self.probas = probas if probas else None
        self.validate_data_on_predict = validate_data_on_predict

        if self.predictions is not None:
            self.predict = self._predict

        if self.probas is not None:
            self.predict_proba = self._predict_proba

    def _predict(self, data: TextData) -> TTextPred:
        """Predict on given data by the data indexes."""
        return list(itemgetter(*data.index)(self.predictions[data.name]))

    def _predict_proba(self, data: TextData) -> TTextProba:
        """Predict probabilities on given data by the data indexes."""
        return list(itemgetter(*data.index)(self.probas))

    def fit(self, *args, **kwargs):
        """Just for python 3.6 (sklearn validates fit method)."""

    @staticmethod
    def _validate_prediction(dataset: TextData, prediction: TTextPred):
        """Validate prediction for given dataset."""
        classification_format_error = f'Check requires classification for {dataset.name} to be ' \
                                      f'either a sequence that can be cast to a 1D numpy array of shape' \
                                      f' (n_samples,), or a sequence of sequences that can be cast to a 2D ' \
                                      f'numpy array of shape (n_samples, n_classes) for the multilabel case.'

        if dataset.task_type == TaskType.TEXT_CLASSIFICATION:
            try:
                prediction = np.array(prediction, dtype='float')
                if not dataset.is_multilabel:
                    prediction = prediction.reshape((-1, 1))
            except ValueError as e:
                raise ValidationError(classification_format_error) from e
            pred_shape = prediction.shape
            n_classes = dataset.num_classes
            if dataset.is_multilabel and len(pred_shape) != 1 and pred_shape[1] != n_classes:
                raise ValidationError(classification_format_error)
            if pred_shape[0] != dataset.n_samples:
                raise ValidationError(f'Check requires classification predictions for {dataset.name} dataset '
                                      f'to have {dataset.n_samples} rows, same as dataset')
            if dataset.is_multilabel:
                if not np.array_equal(prediction, prediction.astype(bool)):
                    raise ValidationError(f'Check requires classification predictions for {dataset.name} dataset '
                                          f'to be either 0 or 1')
        elif dataset.task_type == TaskType.TOKEN_CLASSIFICATION:
            if not isinstance(prediction, collections.abc.Sequence):
                ValidationError(f'Check requires token classification for {dataset.name} to be a sequence')
            if len(prediction) != dataset.n_samples:
                raise ValidationError(f'Check requires token classification for {dataset.name} to have '
                                      f'{dataset.n_samples} rows, same as dataset')
            if not all(isinstance(pred, collections.abc.Sequence) for pred in prediction):
                raise ValidationError(f'Check requires token classification for {dataset.name} to be a sequence '
                                      f'of sequences')
            for sample_pred in prediction:
                for token_pred in sample_pred:
                    if len(token_pred) != 4:
                        raise ValidationError(f'Check requires token classification for {dataset.name} to have '
                                              f'4 entries')
                    if dataset.has_label() and (token_pred[0] not in dataset.classes):
                        raise ValidationError(f'Check requires token classification for {dataset.name} to have '
                                              f'classes in {dataset.classes}, which are the labels in the dataset. '
                                              f'Found class {token_pred[0]}. Classes are defined by the first entry in '
                                              f'the token prediction tuples')
                    if not isinstance(token_pred[1], int) or not isinstance(token_pred[2], int):
                        raise ValidationError(f'Check requires token classification for {dataset.name} to have '
                                              f'int indices representing the start and end of the token, at the second'
                                              f'and third entry in the token prediction tuples')
                    if not isinstance(token_pred[3], float) or (token_pred[3] < 0. or token_pred[3] > 1.):
                        raise ValidationError(f'Check requires token classification for {dataset.name} to have '
                                              f'probabilities between 0 and 1, at the fourth entry in the token '
                                              f'prediction tuples')

    @staticmethod
    def _validate_proba(dataset: TextData, prediction: TTextProba,
                        eps: float = 1e-3):
        """Validate predicted probabilites for given dataset."""
        classification_format_error = f'Check requires classification probabilities for {dataset.name} to be a ' \
                                      f'sequence of sequences that can be cast to a 2D numpy array of shape' \
                                      f' (n_samples, n_classes)'
        if dataset.task_type == TaskType.TEXT_CLASSIFICATION:
            try:
                prediction = np.ndarray(prediction, dtype='float')
            except ValueError as e:
                raise ValidationError(classification_format_error) from e
            pred_shape = prediction.shape
            if len(pred_shape) != 2:
                raise ValidationError(classification_format_error)
            n_classes = dataset.num_classes
            if pred_shape[1] != n_classes:
                raise ValidationError(f'Check requires classification probabilities for {dataset.name} dataset '
                                      f'to have {n_classes} columns, same as the number of classes')
            if pred_shape[0] != dataset.n_samples:
                raise ValidationError(f'Check requires classification probabilities for {dataset.name} dataset '
                                      f'to have {dataset.n_samples} rows, same as dataset')
            if dataset.is_multilabel:
                if (prediction > 1).any() or (prediction < 0).any():
                    raise ValidationError(f'Check requires classification probabilities for {dataset.name} '
                                          f'dataset to be between 0 and 1')
            else:
                if any(abs(prediction.sum(dim=1) - 1) > eps):
                    raise ValidationError(f'Check requires classification probabilities for {dataset.name} '
                                          f'dataset to be probabilities and sum to 1 for each row')


class Context:
    """Contains all the data + properties the user has passed to a check/suite, and validates it seamlessly.

    Parameters
    ----------
    train_dataset: Union[TextData, None] , default: None
        TextData object, representing data an estimator was fitted on
    test_dataset: Union[TextData, None] , default: None
        TextData object, representing data an estimator predicts on
    with_display : bool , default: True
        flag that determines if checks will calculate display (redundant in some checks).
    train_pred: Union[TTextPred, None] , default: None
        predictions on train dataset
    test_pred: Union[TTextPred, None] , default: None
        predictions on test dataset
    train_proba: Union[TTextProba, None] , default: None
        probabilities on train dataset
    test_proba: Union[TTextProba, None] , default: None
        probabilities on test dataset
    """

    def __init__(
        self,
        train_dataset: t.Union[TextData, None] = None,
        test_dataset: t.Union[TextData, None] = None,
        with_display: bool = True,
        train_pred: t.Optional[TTextPred] = None,
        test_pred: t.Optional[TTextPred] = None,
        train_proba: t.Optional[TTextProba] = None,
        test_proba: t.Optional[TTextProba] = None
    ):
        # Validations
        if train_dataset is not None:
            train_dataset = TextData.cast_to_dataset(train_dataset)
        if test_dataset is not None:
            test_dataset = TextData.cast_to_dataset(test_dataset)
        # If both dataset, validate they fit each other
        if train_dataset and test_dataset:
            if test_dataset.has_label() and train_dataset.has_label() and not \
                    TextData.datasets_share_label(train_dataset, test_dataset):
                raise DatasetValidationError('train_dataset and test_dataset must share the same label and task type')
            if train_dataset.name == test_dataset.name:
                raise DatasetValidationError('train_dataset and test_dataset must have different names')
        if test_dataset and not train_dataset:
            raise DatasetValidationError('Can\'t initialize context with only test_dataset. if you have single '
                                         'dataset, initialize it as train_dataset')

        if any(x is not None for x in (train_pred, test_pred, train_proba, test_proba)):
            self._model = _DummyModel(train=train_dataset, test=test_dataset,
                                      y_pred_train=train_pred, y_pred_test=test_pred,
                                      y_proba_test=train_proba, y_proba_train=test_proba)
        else:
            self._model = None

        self._train = train_dataset
        self._test = test_dataset
        self._validated_model = False
        self._task_type = None
        self._with_display = with_display

    @property
    def with_display(self) -> bool:
        """Return the with_display flag."""
        return self._with_display

    @property
    def train(self) -> TextData:
        """Return train_dataset if exists, otherwise raise error."""
        if self._train is None:
            raise DeepchecksNotSupportedError('Check is irrelevant for Datasets without train_dataset dataset')
        return self._train

    @property
    def test(self) -> TextData:
        """Return test_dataset if exists, otherwise raise error."""
        if self._test is None:
            raise DeepchecksNotSupportedError('Check is irrelevant for Datasets without test_dataset dataset')
        return self._test

    @property
    def model(self) -> _DummyModel:
        """Return model if exists, otherwise raise error."""
        if self._model is None:
            raise DeepchecksNotSupportedError('Check is irrelevant without providing predictions')
        return self._model

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

    def get_data_by_kind(self, kind: DatasetKind):
        """Return the relevant Dataset by given kind."""
        if kind == DatasetKind.TRAIN:
            return self.train
        elif kind == DatasetKind.TEST:
            return self.test
        else:
            raise DeepchecksValueError(f'Unexpected dataset kind {kind}')

    def finalize_check_result(self, check_result, check, kind: DatasetKind = None):
        """Run final processing on a check result which includes validation, conditions processing and sampling\
        footnote."""
        # Validate the check result type
        if isinstance(check_result, CheckFailure):
            return
        if not isinstance(check_result, CheckResult):
            raise DeepchecksValueError(f'Check {check.name()} expected to return CheckResult but got: '
                                       + type(check_result).__name__)

        # Set reference between the check result and check
        check_result.check = check
        # Calculate conditions results
        check_result.process_conditions()
        # Add sampling footnote if needed
        if hasattr(check, 'n_samples'):
            n_samples = getattr(check, 'n_samples')
            message = ''
            if kind:
                dataset = self.get_data_by_kind(kind)
                if dataset.is_sampled(n_samples):
                    message = f'Data is sampled from the original dataset, running on ' \
                              f'{dataset.len_when_sampled(n_samples)} samples out of {len(dataset)}.'
            else:
                if self._train is not None and self._train.is_sampled(n_samples):
                    message += f'Running on {self._train.len_when_sampled(n_samples)} <b>train_dataset</b> data samples ' \
                               f'out of {len(self._train)}.'
                if self._test is not None and self._test.is_sampled(n_samples):
                    if message:
                        message += ' '
                    message += f'Running on {self._test.len_when_sampled(n_samples)} <b>test_dataset</b> data samples ' \
                               f'out of {len(self._test)}.'

            if message:
                message = ('<p style="font-size:0.9em;line-height:1;"><i>'
                           f'Note - data sampling: {message} Sample size can be controlled with the "n_samples" '
                           'parameter.</i></p>')
                check_result.display.append(message)

    def get_scorers(self,
                    scorers: t.Union[t.Mapping[str, t.Union[str, t.Callable]], t.List[str]] = None,
                    use_avg_defaults=True) -> t.List[DeepcheckScorer]:
        """Return initialized & validated scorers in a given priority.

        If receive `scorers` use them,
        Else if user defined global scorers use them,
        Else use default scorers.

        Parameters
        ----------
        scorers : Union[List[str], Dict[str, Union[str, Callable]]], default: None
            List of scorers to use. If None, use default scorers.
            Scorers can be supplied as a list of scorer names or as a dictionary of names and functions.
        use_avg_defaults : bool, default True
            If no scorers were provided, for classification, determines whether to use default scorers that return
            an averaged metric, or default scorers that return a metric per class.
        Returns
        -------
        List[DeepcheckScorer]
            A list of initialized & validated scorers.
        """
        if self.task_type == TaskType.TEXT_CLASSIFICATION:
            scorers = scorers or get_default_scorers(TabularTaskType.MULTICLASS, use_avg_defaults)
        elif self.task_type == TaskType.TOKEN_CLASSIFICATION:
            scorers = []  # TODO: Complete that
        else:
            raise DeepchecksValueError(f'Task type must be either {TaskType.TEXT_CLASSIFICATION} or '
                                       f'{TaskType.TOKEN_CLASSIFICATION} but received {self.task_type}')
        return init_validate_scorers(scorers)
