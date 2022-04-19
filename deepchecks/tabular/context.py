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
from typing import Callable, Union, Mapping, Optional

import pandas as pd

from deepchecks.tabular.dataset import Dataset
from deepchecks.tabular.utils.validation import validate_model, model_type_validation
from deepchecks.utils.metrics import ModelType, task_type_check, get_default_scorers, init_validate_scorers
from deepchecks.utils.typing import BasicModel
from deepchecks.utils.features import calculate_feature_importance_or_none
from deepchecks.core.errors import (
    DatasetValidationError, ModelValidationError,
    DeepchecksNotSupportedError, DeepchecksValueError
)


__all__ = [
    'Context'
]


class Context:
    """Contains all the data + properties the user has passed to a check/suite, and validates it seamlessly.

    Parameters
    ----------
    train : Union[Dataset, pd.DataFrame] , default: None
        Dataset or DataFrame object, representing data an estimator was fitted on
    test : Union[Dataset, pd.DataFrame] , default: None
        Dataset or DataFrame object, representing data an estimator predicts on
    model : BasicModel , default: None
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
    """

    def __init__(self,
                 train: Union[Dataset, pd.DataFrame] = None,
                 test: Union[Dataset, pd.DataFrame] = None,
                 model: BasicModel = None,
                 model_name: str = '',
                 features_importance: pd.Series = None,
                 feature_importance_force_permutation: bool = False,
                 feature_importance_timeout: int = 120,
                 scorers: Mapping[str, Union[str, Callable]] = None,
                 scorers_per_class: Mapping[str, Union[str, Callable]] = None
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
    def features_importance(self) -> Optional[pd.Series]:
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
    def features_importance_type(self) -> Optional[str]:
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

    def get_scorers(self, alternative_scorers: Mapping[str, Union[str, Callable]] = None, class_avg=True):
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

    def get_single_scorer(self, alternative_scorers: Mapping[str, Union[str, Callable]] = None, class_avg=True):
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
