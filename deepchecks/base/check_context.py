import abc
from typing import Callable, Union, Mapping, Sequence

import pandas as pd

from deepchecks import BaseCheck, CheckResult, Dataset
from deepchecks.utils.validation import validate_model
from errors import DatasetValidationError, ModelValidationError
from utils.metrics import ModelType, task_type_check, get_scorers_or_default, get_single_scorer_or_default


class CheckRunContext:

    def __init__(self,
                 train=None,
                 test=None,
                 model=None,
                 feature_importance_mode: str = 'model',
                 scorers: Mapping[str, Union[str, Callable]] = None,
                 non_avg_scorers: Mapping[str, Union[str, Callable]] = None
                 ):
        # Validations
        if train and isinstance(train, pd.DataFrame):
            train = Dataset(train, cat_features=[])
        if test and isinstance(test, pd.DataFrame):
            test = Dataset(test, cat_features=[])
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

        if train and model:
            validate_model(train, model)

        self._train = train
        self._test = test
        self._model = model
        self._feature_importance_mode = feature_importance_mode
        self._features_importance = None
        self._task_type = None
        self._user_scorers = scorers
        self._user_non_avg_scorers = non_avg_scorers

    # Properties
    # Validations note: We know train & test fit each other so all validations can be run only on train

    @property
    def train(self):
        if self._train is None:
            raise DatasetValidationError('Check is irrelevant for Datasets without train dataset')
        return self._train

    @property
    def test(self):
        if self._test is None:
            raise DatasetValidationError('Check is irrelevant for Datasets without test dataset')
        return self._test

    @property
    def model(self):
        if self._model is None:
            raise DatasetValidationError('Check is irrelevant for Datasets without model')

    @property
    def label_name(self):
        if self.train.label_name is None:
            raise DatasetValidationError('Check is irrelevant for Datasets without label')
        return self.train.label_name

    @property
    def features(self):
        if not self.train.features_columns:
            raise DatasetValidationError('Check is irrelevant for Datasets without features')
        return self.train.features

    @property
    def date_name(self):
        if self.train.datetime_name is None:
            raise DatasetValidationError('Check is irrelevant for Datasets without datetime column')
        return self.train.datetime_name

    @property
    def index(self):
        if self.train.index_name is None:
            raise DatasetValidationError('Check is irrelevant for Datasets without an index')
        return self.train.index_name

    @property
    def task_type(self):
        if self._task_type is None:
            self._task_type = task_type_check(self.model, self.train)
        return self._task_type

    @property
    def features_importance(self):
        if self._features_importance is None:
            if self._feature_importance_mode == 'model':
                self._features_importance = 0  # TODO get feature importance from model
            elif self._feature_importance_mode == 'permutation':
                self._features_importance = 0  # TODO get feature importance permutation
            else:
                raise DatasetValidationError(f'Unsupported feature importance mode: {self._feature_importance_mode}')
        return self._features_importance

    def assert_task_type(self, expected_types: Sequence[ModelType]):
        if self.task_type not in expected_types:
            raise ModelValidationError(
                f'Check is relevant for models of type {[e.value.lower() for e in expected_types]}, '
                f"but received model of type '{self.task_type.value.lower()}'"  # pylint: disable=inconsistent-quotes
            )

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


class SingleDatasetBaseCheck(BaseCheck):
    """Parent class for checks that only use one dataset."""

    def run(self, dataset, model=None) -> CheckResult:
        # By default, we initialize a single dataset as the "train"
        c = CheckRunContext(dataset, model=model)
        return self.run_logic(c)

    @abc.abstractmethod
    def run_logic(self, context: CheckRunContext, dataset: str = 'train'):
        pass


class TrainTestBaseCheck(BaseCheck):
    """Parent class for checks that compare two datasets.

    The class checks train dataset and test dataset for model training and test.
    """

    def run(self, train_dataset, test_dataset, model=None) -> CheckResult:
        c = CheckRunContext(train_dataset, test_dataset, model=model)
        return self.run_logic(c)

    @abc.abstractmethod
    def run_logic(self, context: CheckRunContext):
        pass


class ModelOnlyBaseCheck(BaseCheck):
    """Parent class for checks that only use a model and no datasets."""

    def run(self, model) -> CheckResult:
        """Define run signature."""
        c = CheckRunContext(model=model)
        return self.run_logic(c)

    @abc.abstractmethod
    def run_logic(self, context: CheckRunContext):
        pass
