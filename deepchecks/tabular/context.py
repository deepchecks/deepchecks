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
"""Module for base tabular context."""
import typing as t
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

from deepchecks.core import DatasetKind
from deepchecks.core.errors import (DatasetValidationError,
                                    DeepchecksNotSupportedError,
                                    DeepchecksValueError, ModelValidationError)
from deepchecks.tabular.dataset import Dataset
from deepchecks.tabular.utils.validation import (model_type_validation,
                                                 validate_model)
from deepchecks.utils.features import calculate_feature_importance_or_none
from deepchecks.utils.metrics import (ModelType, get_default_scorers,
                                      init_validate_scorers, task_type_check)
from deepchecks.utils.typing import BasicModel

__all__ = [
    'Context'
]


class _DummyModel:
    """Contains all the data + properties the user has passed to a check/suite, and validates it seamlessly.

    Parameters
    ----------
    train: Dataset
        Dataset, representing data an estimator was fitted on
    test: Dataset
        Dataset, representing data an estimator predicts on
    y_pred_train: np.ndarray
        Array of the model prediction over the train dataset.
    y_pred_test: np.ndarray
        Array of the model prediction over the test dataset.
    y_proba_train: np.ndarray
        Array of the model prediction probabilities over the train dataset.
    y_proba_test: np.ndarray
        Array of the model prediction probabilities over the test dataset.
    """

    train: pd.DataFrame
    test: pd.DataFrame
    ind_map: t.Dict[str, t.Dict[t.Any, int]]
    y_pred: t.Dict[str, np.ndarray]
    y_proba: t.Dict[str, np.ndarray]

    def __init__(self,
                 train: Dataset,
                 test: Dataset,
                 y_pred_train: np.ndarray,
                 y_pred_test: np.ndarray,
                 y_proba_train: np.ndarray,
                 y_proba_test: np.ndarray,):
        self.ind_map = defaultdict(dict)

        if train is not None and test is not None:
            # check if datasets have same indexes
            if set(train.data.index) & set(test.data.index):
                train.data.index = map(lambda x: f'train-{x}', list(train.data.index))
                test.data.index = map(lambda x: f'test-{x}', list(test.data.index))
                warnings.warn('train and test datasets have common index - adding "train"/"test"'
                              ' prefixes. To avoid that provide datasets with no common indexes '
                              'or pass the model object instead of the predictions.')

        if train is not None:
            self.train = train.features_columns
            for i, ind in enumerate(self.train.index):
                self.ind_map[DatasetKind.TRAIN][ind] = i

        if test is not None:
            self.test = test.features_columns
            for i, ind in enumerate(self.test.index):
                self.ind_map[DatasetKind.TEST][ind] = i

        if y_pred_train is not None or y_pred_test is not None:
            self.y_pred = {}
            self.y_pred[DatasetKind.TRAIN] = y_pred_train
            self.y_pred[DatasetKind.TEST] = y_pred_test
            self.predict = self._predict

        if y_proba_train is not None or y_proba_train is not None:
            self.y_proba = {}
            self.y_proba[DatasetKind.TRAIN] = y_proba_train
            self.y_proba[DatasetKind.TEST] = y_proba_test
            self.predict_proba = self._predict_proba

    def _get_dataset(self, dataset_kind: DatasetKind):
        if dataset_kind == DatasetKind.TRAIN:
            return self.train
        if dataset_kind == DatasetKind.TEST:
            return self.test
        raise DeepchecksValueError(f'Unexpected dataset kind {dataset_kind}')

    def _validate_data(self, data: pd.DataFrame, dataset_kind: DatasetKind):
        # validate that the data isn't a new data by compairing a sample of rows.
        sub_index = list(data.index)[:10]
        return (data.loc[sub_index].fillna('nan') ==
                self._get_dataset(dataset_kind).loc[sub_index].fillna('nan')).all().all()

    def _get_dataset_kind(self, data: pd.DataFrame):
        assert isinstance(data, pd.DataFrame)
        if set(data.index).issubset(list(self.ind_map[DatasetKind.TRAIN].keys())):
            dataset_kind = DatasetKind.TRAIN
        elif set(data.index).issubset(list(self.ind_map[DatasetKind.TEST].keys())):
            dataset_kind = DatasetKind.TEST
        else:
            dataset_kind = None
        if dataset_kind is None or not self._validate_data(data, dataset_kind):
            raise DeepchecksValueError('New data recieved for static model predictions.')  # TODO write something cooler
        return dataset_kind

    def _get_index_to_predict(self, data: pd.DataFrame):
        dataset_kind = self._get_dataset_kind(data)
        return dataset_kind, [self.ind_map[dataset_kind][i] for i in data.index]

    def _predict(self, data: pd.DataFrame):
        dataset_kind, indexes_to_predict = self._get_index_to_predict(data)
        return self.y_pred[dataset_kind][indexes_to_predict]

    def _predict_proba(self, data: pd.DataFrame):
        dataset_kind, indexes_to_predict = self._get_index_to_predict(data)
        return self.y_proba[dataset_kind][indexes_to_predict]


class Context:
    """Contains all the data + properties the user has passed to a check/suite, and validates it seamlessly.

    Parameters
    ----------
    train: Union[Dataset, pd.DataFrame] , default: None
        Dataset or DataFrame object, representing data an estimator was fitted on
    test: Union[Dataset, pd.DataFrame] , default: None
        Dataset or DataFrame object, representing data an estimator predicts on
    model: BasicModel , default: None
        A scikit-learn-compatible fitted estimator instance
    model_name: str , default: ''
        The name of the model
    features_importance: pd.Series , default: None
        pass manual features importance
    feature_importance_force_permutation : bool , default: False
        force calculation of permutation features importance
    feature_importance_timeout : int , default: 120
        timeout in second for the permutation features importance calculation
    scorers : Mapping[str, Union[str, Callable]] , default: None
        dict of scorers names to scorer sklearn_name/function
    scorers_per_class : Mapping[str, Union[str, Callable]] , default: None
        dict of scorers for classification without averaging of the classes.
        See <a href=
        "https://scikit-learn.org/stable/modules/model_evaluation.html#from-binary-to-multiclass-and-multilabel">
        scikit-learn docs</a>
    y_pred_train: np.ndarray , default: None
        Array of the model prediction over the train dataset.
    y_pred_test: np.ndarray , default: None
        Array of the model prediction over the test dataset.
    y_proba_train: np.ndarray , default: None
        Array of the model prediction probabilities over the train dataset.
    y_proba_test: np.ndarray , default: None
        Array of the model prediction probabilities over the test dataset.
    """

    def __init__(self,
                 train: t.Union[Dataset, pd.DataFrame] = None,
                 test: t.Union[Dataset, pd.DataFrame] = None,
                 model: BasicModel = None,
                 model_name: str = '',
                 features_importance: pd.Series = None,
                 feature_importance_force_permutation: bool = False,
                 feature_importance_timeout: int = 120,
                 scorers: t.Mapping[str, t.Union[str, t.Callable]] = None,
                 scorers_per_class: t.Mapping[str, t.Union[str, t.Callable]] = None,
                 y_pred_train: np.ndarray = None,
                 y_pred_test: np.ndarray = None,
                 y_proba_train: np.ndarray = None,
                 y_proba_test: np.ndarray = None,
                 ):
        # Validations
        if train is None and test is None and model is None:
            raise DeepchecksValueError('At least one dataset (or model) must be passed to the method!')
        if train is not None:
            train = Dataset.cast_to_dataset(train)
        if test is not None:
            test = Dataset.cast_to_dataset(test)
        # If both dataset, validate they fit each other
        if train and test:
            if not Dataset.datasets_share_label(train, test):
                raise DatasetValidationError('train and test requires to have and to share the same label')
            if not Dataset.datasets_share_features(train, test):
                raise DatasetValidationError('train and test requires to share the same features columns')
            if not Dataset.datasets_share_categorical_features(train, test):
                raise DatasetValidationError(
                    'train and test datasets should share '
                    'the same categorical features. Possible reason is that some columns were'
                    'inferred incorrectly as categorical features. To fix this, manually edit the '
                    'categorical features using Dataset(cat_features=<list_of_features>'
                )
            if not Dataset.datasets_share_index(train, test):
                raise DatasetValidationError('train and test requires to share the same index column')
            if not Dataset.datasets_share_date(train, test):
                raise DatasetValidationError('train and test requires to share the same date column')
        if test and not train:
            raise DatasetValidationError('Can\'t initialize context with only test. if you have single dataset, '
                                         'initialize it as train')
        if model is None and \
           not pd.Series([y_pred_train, y_pred_test, y_proba_train, y_proba_test]).isna().all():
            model = _DummyModel(train=train, test=test, y_pred_train=y_pred_train, y_pred_test=y_pred_test,
                                y_proba_test=y_proba_test, y_proba_train=y_proba_train)
        if model is not None:
            # Here validate only type of model, later validating it can predict on the data if needed
            model_type_validation(model)
        if features_importance is not None:
            if not isinstance(features_importance, pd.Series):
                raise DeepchecksValueError('features_importance must be a pandas Series')
        self._train = train
        self._test = test
        self._model = model
        self._feature_importance_force_permutation = feature_importance_force_permutation
        self._features_importance = features_importance
        self._feature_importance_timeout = feature_importance_timeout
        self._calculated_importance = features_importance is not None
        self._importance_type = None
        self._validated_model = False
        self._task_type = None
        self._user_scorers = scorers
        self._user_scorers_per_class = scorers_per_class
        self._model_name = model_name

    # Properties
    # Validations note: We know train & test fit each other so all validations can be run only on train

    @property
    def train(self) -> Dataset:
        """Return train if exists, otherwise raise error."""
        if self._train is None:
            raise DeepchecksNotSupportedError('Check is irrelevant for Datasets without train dataset')
        return self._train

    @property
    def test(self) -> Dataset:
        """Return test if exists, otherwise raise error."""
        if self._test is None:
            raise DeepchecksNotSupportedError('Check is irrelevant for Datasets without test dataset')
        return self._test

    @property
    def model(self) -> BasicModel:
        """Return & validate model if model exists, otherwise raise error."""
        if self._model is None:
            raise DeepchecksNotSupportedError('Check is irrelevant for Datasets without model')
        if not self._validated_model:
            if self._train:
                validate_model(self._train, self._model)
            self._validated_model = True
        return self._model

    @property
    def model_name(self):
        """Return model name."""
        return self._model_name

    @property
    def task_type(self) -> ModelType:
        """Return task type if model & train & label exists. otherwise, raise error."""
        if self._task_type is None:
            self._task_type = task_type_check(self.model, self.train)
        return self._task_type

    @property
    def features_importance(self) -> t.Optional[pd.Series]:
        """Return features importance, or None if not possible."""
        if not self._calculated_importance:
            if self._model and (self._train or self._test):
                permutation_kwargs = {'timeout': self._feature_importance_timeout}
                dataset = self.test if self.have_test() else self.train
                importance, importance_type = calculate_feature_importance_or_none(
                    self._model, dataset, self._feature_importance_force_permutation, permutation_kwargs
                )
                self._features_importance = importance
                self._importance_type = importance_type
            else:
                self._features_importance = None
            self._calculated_importance = True

        return self._features_importance

    @property
    def features_importance_type(self) -> t.Optional[str]:
        """Return feature importance type if feature importance is available, else None."""
        # Calling first feature_importance, because _importance_type is assigned only after feature importance is
        # calculated.
        if self.features_importance:
            return self._importance_type
        return None

    def have_test(self):
        """Return whether there is test dataset defined."""
        return self._test is not None

    def assert_task_type(self, *expected_types: ModelType):
        """Assert task_type matching given types.

        If task_type is defined, validate it and raise error if needed, else returns True.
        If task_type is not defined, return False.
        """
        # To calculate task type we need model and train. if not exists return False, means we did not validate
        if self._model is None or self._train is None:
            return False
        if self.task_type not in expected_types:
            raise ModelValidationError(
                f'Check is relevant for models of type {[e.value.lower() for e in expected_types]}, '
                f"but received model of type '{self.task_type.value.lower()}'"  # pylint: disable=inconsistent-quotes
            )
        return True

    def assert_classification_task(self):
        """Assert the task_type is classification."""
        # assert_task_type makes assertion if task type exists and returns True, else returns False
        # If not task type than check label type
        if (not self.assert_task_type(ModelType.MULTICLASS, ModelType.BINARY) and
                self.train.label_type == 'regression_label'):
            raise ModelValidationError('Check is irrelevant for regressions tasks')

    def assert_regression_task(self):
        """Assert the task type is regression."""
        # assert_task_type makes assertion if task type exists and returns True, else returns False
        # If not task type than check label type
        if (not self.assert_task_type(ModelType.REGRESSION) and
                self.train.label_type == 'classification_label'):
            raise ModelValidationError('Check is irrelevant for classification tasks')

    def get_scorers(self, alternative_scorers: t.Mapping[str, t.Union[str, t.Callable]] = None, class_avg=True):
        """Return initialized & validated scorers in a given priority.

        If receive `alternative_scorers` return them,
        Else if user defined global scorers return them,
        Else return default scorers.

        Parameters
        ----------
        alternative_scorers : Mapping[str, Union[str, Callable]], default None
            dict of scorers names to scorer sklearn_name/function
        class_avg : bool, default True
            for classification whether to return scorers of average score or score per class
        """
        if class_avg:
            user_scorers = self._user_scorers
        else:
            user_scorers = self._user_scorers_per_class

        scorers = alternative_scorers or user_scorers or get_default_scorers(self.task_type, class_avg)
        return init_validate_scorers(scorers, self.model, self.train, class_avg, self.task_type)

    def get_single_scorer(self, alternative_scorers: t.Mapping[str, t.Union[str, t.Callable]] = None, class_avg=True):
        """Return initialized & validated single scorer in a given priority.

        If receive `alternative_scorers` use them,
        Else if user defined global scorers use them,
        Else use default scorers.
        Returns the first scorer from the scorers described above.

        Parameters
        ----------
        alternative_scorers : Mapping[str, Union[str, Callable]], default None
            dict of scorers names to scorer sklearn_name/function. Only first scorer will be used.
        class_avg : bool, default True
            for classification whether to return scorers of average score or score per class
        """
        if class_avg:
            user_scorers = self._user_scorers
        else:
            user_scorers = self._user_scorers_per_class

        scorers = alternative_scorers or user_scorers or get_default_scorers(self.task_type, class_avg)
        # The single scorer is the first one in the dict
        scorer_name = next(iter(scorers))
        single_scorer_dict = {scorer_name: scorers[scorer_name]}
        return init_validate_scorers(single_scorer_dict, self.model, self.train, class_avg, self.task_type)[0]

    def get_data_by_kind(self, kind: DatasetKind):
        """Return the relevant Dataset by given kind."""
        if kind == DatasetKind.TRAIN:
            return self.train
        elif kind == DatasetKind.TEST:
            return self.test
        else:
            raise DeepchecksValueError(f'Unexpected dataset kind {kind}')

    def get_is_sampled_footnote(self, n_samples: int, kind: DatasetKind = None):
        """Get footnote to display when the datasets are sampled."""
        message = ''
        if kind:
            v_data = self.get_data_by_kind(kind)
            if v_data.is_sampled(n_samples):
                message = f'Data is sampled from the original dataset, running on ' \
                          f'{v_data.len_when_sampled(n_samples)} samples out of {len(v_data)}.'
        else:
            if self._train is not None and self._train.is_sampled(n_samples):
                message += f'Running on {self._train.len_when_sampled(n_samples)} <b>train</b> data samples out of ' \
                           f'{len(self._train)}.'
            if self._test is not None and self._test.is_sampled(n_samples):
                if message:
                    message += ' '
                message += f'Running on {self._test.len_when_sampled(n_samples)} <b>test</b> data samples out of ' \
                           f'{len(self._test)}.'

        if message:
            message = f'Note - data sampling: {message} Sample size can be controlled with the "n_samples" parameter.'
            return f'<p style="font-size:0.9em;line-height:1;"><i>{message}</i></p>'
