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

import numpy as np

from deepchecks.nlp.task_type import TaskType
from deepchecks.nlp.text_data import TextData

from deepchecks.core import CheckFailure, CheckResult, DatasetKind
from deepchecks.core.errors import (DatasetValidationError, DeepchecksNotSupportedError, DeepchecksValueError,
                                    ModelValidationError, ValidationError)


__all__ = [
    'Context',
    'TTextPred'
]


TClassPred = t.Sequence[t.Sequence[float]]
TTokenPred = t.Sequence[t.Sequence[t.Tuple[str, int, int, float]]]
TTextPred = t.Union[TClassPred, TTokenPred]


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
    train_predictions: Union[TTextPred, None] , default: None
        predictions on train dataset
    test_predictions: Union[TTextPred, None] , default: None
        predictions on test dataset
    """

    def __init__(
        self,
        train_dataset: t.Union[TextData, None] = None,
        test_dataset: t.Union[TextData, None] = None,
        with_display: bool = True,
        train_predictions: t.Optional[TTextPred] = None,
        test_predictions: t.Optional[TTextPred] = None,
    ):
        # Validations
        if train_dataset is not None:
            train_dataset = TextData.cast_to_dataset(train_dataset)
        if test_dataset is not None:
            test_dataset = TextData.cast_to_dataset(test_dataset)
        # If both dataset, validate they fit each other
        if train_dataset and test_dataset:
            if test_dataset.has_label() and train_dataset.has_label() and not TextData.datasets_share_label(train_dataset, test_dataset):
                raise DatasetValidationError('train_dataset and test_dataset must share the same label and task type')
        if test_dataset and not train_dataset:
            raise DatasetValidationError('Can\'t initialize context with only test_dataset. if you have single dataset, '
                                         'initialize it as train_dataset')

        if train_predictions is not None or test_predictions is not None:
            self._static_predictions = {}
            for dataset, dataset_type, predictions in zip([train_dataset, test_dataset],
                                                          [DatasetKind.TRAIN, DatasetKind.TEST],
                                                          [train_predictions, test_predictions]):
                if dataset is not None:
                    self._validate_prediction(dataset, dataset_type, predictions)
                    self._static_predictions[dataset_type] = predictions

        self._train = train_dataset
        self._test = test_dataset
        self._validated_model = False
        self._task_type = None
        self._with_display = with_display

    # Validations note: We know train_dataset & test_dataset fit each other so all validations can be run only on train_dataset

    @staticmethod
    def _validate_prediction(dataset: TextData, dataset_type: DatasetKind, prediction: TTextPred,
                             eps: float = 1e-3):
        """Validate prediction for given dataset."""
        classification_format_error = f'Check requires classification for {dataset_type} to be a sequence '\
                                      f'of sequences that can be cast to a 2D numpy array of shape'\
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
                raise ValidationError(f'Check requires classification predictions for {dataset_type.value} dataset '
                                      f'to have {n_classes} columns, same as the number of classes')
            if pred_shape[0] != dataset.n_samples:
                raise ValidationError(f'Check requires classification predictions for {dataset_type.value} dataset '
                                      f'to have {dataset.n_samples} rows, same as dataset')
            if dataset.is_multilabel:
                if (prediction > 1).any() or (prediction < 0).any():
                    raise ValidationError(f'Check requires classification predictions for {dataset_type.value} dataset '
                                          f'to be between 0 and 1')
            else:
                if any(abs(prediction.sum(dim=1) - 1) > eps):
                    raise ValidationError(f'Check requires classification predictions for {dataset_type.value} dataset '
                                          f'to be probabilities and sum to 1 for each row')
        elif dataset.task_type == TaskType.TOKEN_CLASSIFICATION:
            if not isinstance(prediction, collections.abc.Sequence):
                ValidationError(f'Check requires token classification for {dataset_type.value} to be a sequence')
            if len(prediction) != dataset.n_samples:
                raise ValidationError(f'Check requires token classification for {dataset_type.value} to have '
                                      f'{dataset.n_samples} rows, same as dataset')
            if not all(isinstance(pred, collections.abc.Sequence) for pred in prediction):
                raise ValidationError(f'Check requires token classification for {dataset_type.value} to be a sequence '
                                      f'of sequences')
            for sample_pred in prediction:
                for token_pred in sample_pred:
                    if len(token_pred) != 4:
                        raise ValidationError(f'Check requires token classification for {dataset_type.value} to have '
                                              f'4 entries')
                    if dataset.has_label() and (token_pred[0] not in dataset.classes):
                        raise ValidationError(f'Check requires token classification for {dataset_type.value} to have '
                                              f'classes in {dataset.classes}, which are the labels in the dataset. '
                                              f'Found class {token_pred[0]}. Classes are defined by the first entry in '
                                              f'the token prediction tuples')
                    if not isinstance(token_pred[1], int) or not isinstance(token_pred[2], int):
                        raise ValidationError(f'Check requires token classification for {dataset_type.value} to have '
                                              f'int indices representing the start and end of the token, at the second'
                                              f'and third entry in the token prediction tuples')
                    if not isinstance(token_pred[3], float) or (token_pred[3] < 0. or token_pred[3] > 1.):
                        raise ValidationError(f'Check requires token classification for {dataset_type.value} to have '
                                              f'probabilities between 0 and 1, at the fourth entry in the token '
                                              f'prediction tuples')

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

    def get_predictions_by_kind(self, kind: DatasetKind):
        """Return the relevant predictions by given kind."""
        if kind in [DatasetKind.TRAIN, DatasetKind.TEST]:
            return self._static_predictions[kind]
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
