from typing import Callable, Union, Mapping, Sequence, Dict, List, Optional

import pandas as pd

from deepchecks import Dataset
from deepchecks.utils.validation import validate_model
from deepchecks.errors import DatasetValidationError, ModelValidationError
from deepchecks.utils.metrics import ModelType, task_type_check, get_scorers_or_default, get_single_scorer_or_default
from deepchecks.utils.typing import Hashable, BasicModel
from utils.features import calculate_feature_importance_or_none


class CheckRunContext:

    def __init__(self,
                 train: Dataset = None,
                 test: Dataset = None,
                 model: BasicModel = None,
                 features_importance: Dict[Hashable, float] = None,
                 feature_importance_force_permutation: bool = False,
                 feature_importance_timeout: int = None,
                 scorers: Mapping[str, Union[str, Callable]] = None,
                 non_avg_scorers: Mapping[str, Union[str, Callable]] = None
                 ):
        # Validations
        if train:
            train = Dataset.ensure_not_empty_dataset(train, cast=True)
        if test:
            test = Dataset.ensure_not_empty_dataset(test, cast=True)
        # If both datasets, validate they fit each other
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
        if train and model:
            validate_model(train, model)

        self._train = train
        self._test = test
        self._model = model
        self._feature_importance_force_permutation = feature_importance_force_permutation
        self._features_importance = features_importance
        self._feature_importance_timeout = feature_importance_timeout
        self._calculated_importance = False
        self._importance_type = None
        self._task_type = None
        self._user_scorers = scorers
        self._user_non_avg_scorers = non_avg_scorers

    # Properties
    # Validations note: We know train & test fit each other so all validations can be run only on train

    @property
    def train(self) -> Dataset:
        if self._train is None:
            raise DatasetValidationError('Check is irrelevant for Datasets without train dataset')
        return self._train

    @property
    def test(self) -> Dataset:
        if self._test is None:
            raise DatasetValidationError('Check is irrelevant for Datasets without test dataset')
        return self._test

    @property
    def model(self) -> BasicModel:
        if self._model is None:
            raise DatasetValidationError('Check is irrelevant for Datasets without model')

    @property
    def label_name(self) -> str:
        if self.train.label_name is None:
            raise DatasetValidationError('Check is irrelevant for Datasets without label')
        return self.train.label_name

    @property
    def features(self) -> List[Hashable]:
        """Return list of feature names."""
        if not self.train.features:
            raise DatasetValidationError('Check is irrelevant for Datasets without features')
        return self.train.features

    @property
    def cat_features(self) -> List[Hashable]:
        if not self.train.cat_features:
            raise DatasetValidationError('Check is irrelevant for Datasets without categorical features')
        return self.train.cat_features

    @property
    def task_type(self) -> ModelType:
        if self._task_type is None:
            self._task_type = task_type_check(self.model, self.train)
        return self._task_type

    @property
    def features_importance(self) -> Optional[pd.Series]:
        if not self._calculated_importance:
            permutation_kwargs = {}
            if self._feature_importance_timeout:
                permutation_kwargs['timeout'] = self._feature_importance_timeout
            importance, importance_type = calculate_feature_importance_or_none(
                self.model, self.train, self._feature_importance_force_permutation, permutation_kwargs
            )
            self._features_importance = importance
            self._importance_type = importance_type
            self._calculated_importance = True

        return self._features_importance

    @property
    def features_importance_type(self) -> str:
        return self._importance_type

    def assert_task_type(self, *expected_types: ModelType):
        if self.task_type not in expected_types:
            raise ModelValidationError(
                f'Check is relevant for models of type {[e.value.lower() for e in expected_types]}, '
                f"but received model of type '{self.task_type.value.lower()}'"  # pylint: disable=inconsistent-quotes
            )

    def assert_datetime_exists(self):
        if not self.train.datetime_exist():
            raise DatasetValidationError('Check is irrelevant for Datasets without datetime column')

    def assert_index_exists(self):
        if not self.train.datetime_exist():
            raise DatasetValidationError('Check is irrelevant for Datasets without index column')

    def get_scorers(self, alternative_scorers: Mapping[str, Callable] = None, multiclass_non_avg=False):
        if multiclass_non_avg:
            scorers = alternative_scorers or self._user_non_avg_scorers
        else:
            scorers = alternative_scorers or self._user_scorers
        return get_scorers_or_default(self.task_type, self.model, self.train, scorers, multiclass_non_avg)

    def get_single_scorer(self, alternative_scorer: Union[str, Callable], multiclass_non_avg=False):
        if alternative_scorer:
            scorer = alternative_scorer
        else:
            user_scorers = self._user_non_avg_scorers if multiclass_non_avg else self._user_scorers
            if user_scorers:
                scorer_name = next(iter(self._user_non_avg_scorers))
                scorer = self._user_non_avg_scorers[scorer_name]
            else:
                scorer = None
        return get_single_scorer_or_default(self.task_type, self.model, self.train, scorer, multiclass_non_avg)
