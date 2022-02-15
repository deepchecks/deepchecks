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
"""Module for base tabular abstractions."""
# pylint: disable=broad-except
import abc
from functools import wraps
from typing import Callable, Union, Tuple, Mapping, List, Optional, Any

import pandas as pd

from deepchecks.tabular import Dataset
from deepchecks.utils.validation import validate_model, model_type_validation
from deepchecks.utils.metrics import ModelType, task_type_check, get_default_scorers, init_validate_scorers
from deepchecks.utils.typing import BasicModel
from deepchecks.utils.features import calculate_feature_importance_or_none
from deepchecks.core.check import (
    CheckResult,
    BaseCheck,
    CheckFailure,
    SingleDatasetBaseCheck,
    TrainTestBaseCheck,
    ModelOnlyBaseCheck
)
from deepchecks.core.suite import BaseSuite, SuiteResult
from deepchecks.core.display_suite import ProgressBar
from deepchecks.core.errors import (
    DatasetValidationError, ModelValidationError,
    DeepchecksNotSupportedError, DeepchecksValueError
)


__all__ = [
    'Context',
    'Suite',
    'SingleDatasetCheck',
    'TrainTestCheck',
    'ModelOnlyCheck',
    'ModelComparisonSuite',
    'ModelComparisonCheck'
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
            train = Dataset.ensure_not_empty_dataset(train)
        if test is not None:
            test = Dataset.ensure_not_empty_dataset(test)
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

        self._train = train
        self._test = test
        self._model = model
        self._feature_importance_force_permutation = feature_importance_force_permutation
        self._features_importance = features_importance
        self._feature_importance_timeout = feature_importance_timeout
        self._calculated_importance = False
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


def wrap_run(func, class_instance):
    """Wrap the run function of checks, and sets the `check` property on the check result."""
    @wraps(func)
    def wrapped(*args, **kwargs):
        result = func(*args, **kwargs)
        if not isinstance(result, CheckResult):
            raise DeepchecksValueError(f'Check {class_instance.name()} expected to return CheckResult but got: '
                                       + type(result).__name__)
        result.check = class_instance
        result.process_conditions()
        return result

    return wrapped


class SingleDatasetCheck(SingleDatasetBaseCheck):
    """Parent class for checks that only use one dataset."""

    context_type = Context

    def __init__(self):
        super().__init__()
        # Replace the run_logic function with wrapped run function
        setattr(self, 'run_logic', wrap_run(getattr(self, 'run_logic'), self))

    def run(self, dataset, model=None) -> CheckResult:
        """Run check."""
        assert self.context_type is not None
        return self.run_logic(self.context_type(  # pylint: disable=not-callable
            dataset,
            model=model
        ))

    @abc.abstractmethod
    def run_logic(self, context, dataset_type: str = 'train') -> CheckResult:
        """Run check."""
        raise NotImplementedError()


class TrainTestCheck(TrainTestBaseCheck):
    """Parent class for checks that compare two datasets.

    The class checks train dataset and test dataset for model training and test.
    """

    context_type = Context

    def __init__(self):
        super().__init__()
        # Replace the run_logic function with wrapped run function
        setattr(self, 'run_logic', wrap_run(getattr(self, 'run_logic'), self))

    def run(self, train_dataset, test_dataset, model=None) -> CheckResult:
        """Run check."""
        assert self.context_type is not None
        return self.run_logic(self.context_type(  # pylint: disable=not-callable
            train_dataset,
            test_dataset,
            model=model
        ))

    @abc.abstractmethod
    def run_logic(self, context) -> CheckResult:
        """Run check."""
        raise NotImplementedError()


class ModelOnlyCheck(ModelOnlyBaseCheck):
    """Parent class for checks that only use a model and no datasets."""

    context_type = Context

    def __init__(self):
        """Initialize the class."""
        super().__init__()
        # Replace the run_logic function with wrapped run function
        setattr(self, 'run_logic', wrap_run(getattr(self, 'run_logic'), self))

    def run(self, model) -> CheckResult:
        """Run check."""
        assert self.context_type is not None
        return self.run_logic(self.context_type(model=model))  # pylint: disable=not-callable

    @abc.abstractmethod
    def run_logic(self, context) -> CheckResult:
        """Run check."""
        raise NotImplementedError()


class Suite(BaseSuite):
    """Tabular suite to run checks of types: TrainTestCheck, SingleDatasetCheck, ModelOnlyCheck."""

    @classmethod
    def supported_checks(cls) -> Tuple:
        """Return tuple of supported check types of this suite."""
        return TrainTestCheck, SingleDatasetCheck, ModelOnlyCheck

    def run(
            self,
            train_dataset: Optional[Union[Dataset, pd.DataFrame]] = None,
            test_dataset: Optional[Union[Dataset, pd.DataFrame]] = None,
            model: BasicModel = None,
            features_importance: pd.Series = None,
            feature_importance_force_permutation: bool = False,
            feature_importance_timeout: int = None,
            scorers: Mapping[str, Union[str, Callable]] = None,
            scorers_per_class: Mapping[str, Union[str, Callable]] = None
    ) -> SuiteResult:
        """Run all checks.

        Parameters
        ----------
        train_dataset: Optional[Union[Dataset, pd.DataFrame]] , default None
            object, representing data an estimator was fitted on
        test_dataset : Optional[Union[Dataset, pd.DataFrame]] , default None
            object, representing data an estimator predicts on
        model : BasicModel , default None
            A scikit-learn-compatible fitted estimator instance
        features_importance : pd.Series , default None
            pass manual features importance
        feature_importance_force_permutation : bool , default None
            force calculation of permutation features importance
        feature_importance_timeout : int , default None
            timeout in second for the permutation features importance calculation
        scorers : Mapping[str, Union[str, Callable]] , default None
            dict of scorers names to scorer sklearn_name/function
        scorers_per_class : Mapping[str, Union[str, Callable]], default None
            dict of scorers for classification without averaging of the classes
            See <a href=
            "https://scikit-learn.org/stable/modules/model_evaluation.html#from-binary-to-multiclass-and-multilabel">
            scikit-learn docs</a>
        Returns
        -------
        SuiteResult
            All results by all initialized checks
        """
        context = Context(train_dataset, test_dataset, model,
                          features_importance=features_importance,
                          feature_importance_force_permutation=feature_importance_force_permutation,
                          feature_importance_timeout=feature_importance_timeout,
                          scorers=scorers,
                          scorers_per_class=scorers_per_class)
        # Create progress bar
        progress_bar = ProgressBar(self.name, len(self.checks))

        # Run all checks
        results = []
        for check in self.checks.values():
            try:
                progress_bar.set_text(check.name())
                if isinstance(check, TrainTestCheck):
                    if train_dataset is not None and test_dataset is not None:
                        check_result = check.run_logic(context)
                        results.append(check_result)
                    else:
                        msg = 'Check is irrelevant if not supplied with both train and test datasets'
                        results.append(Suite._get_unsupported_failure(check, msg))
                elif isinstance(check, SingleDatasetCheck):
                    if train_dataset is not None:
                        # In case of train & test, doesn't want to skip test if train fails. so have to explicitly
                        # wrap it in try/except
                        try:
                            check_result = check.run_logic(context)
                            # In case of single dataset not need to edit the header
                            if test_dataset is not None:
                                check_result.header = f'{check_result.get_header()} - Train Dataset'
                        except Exception as exp:
                            check_result = CheckFailure(check, exp, ' - Train Dataset')
                        results.append(check_result)
                    if test_dataset is not None:
                        try:
                            check_result = check.run_logic(context, dataset_type='test')
                            # In case of single dataset not need to edit the header
                            if train_dataset is not None:
                                check_result.header = f'{check_result.get_header()} - Test Dataset'
                        except Exception as exp:
                            check_result = CheckFailure(check, exp, ' - Test Dataset')
                        results.append(check_result)
                    if train_dataset is None and test_dataset is None:
                        msg = 'Check is irrelevant if dataset is not supplied'
                        results.append(Suite._get_unsupported_failure(check, msg))
                elif isinstance(check, ModelOnlyCheck):
                    if model is not None:
                        check_result = check.run_logic(context)
                        results.append(check_result)
                    else:
                        msg = 'Check is irrelevant if model is not supplied'
                        results.append(Suite._get_unsupported_failure(check, msg))
                else:
                    raise TypeError(f'Don\'t know how to handle type {check.__class__.__name__} in suite.')
            except Exception as exp:
                results.append(CheckFailure(check, exp))
            progress_bar.inc_progress()

        progress_bar.close()
        return SuiteResult(self.name, results)

    @classmethod
    def _get_unsupported_failure(cls, check, msg):
        return CheckFailure(check, DeepchecksNotSupportedError(msg))


class ModelComparisonSuite(BaseSuite):
    """Suite to run checks of types: CompareModelsBaseCheck."""

    @classmethod
    def supported_checks(cls) -> Tuple:
        """Return tuple of supported check types of this suite."""
        return tuple([ModelComparisonCheck])

    def run(self,
            train_datasets: Union[Dataset, List[Dataset]],
            test_datasets: Union[Dataset, List[Dataset]],
            models: Union[List[Any], Mapping[str, Any]]
            ) -> SuiteResult:
        """Run all checks.

        Parameters
        ----------
        train_datasets : Union[Dataset, Container[Dataset]]
            representing data an estimator was fitted on
        test_datasets: Union[Dataset, Container[Dataset]]
            representing data an estimator was fitted on
        models : Union[Container[Any], Mapping[str, Any]]
            2 or more scikit-learn-compatible fitted estimator instance
        Returns
        -------
        SuiteResult
            All results by all initialized checks
        Raises
        ------
        ValueError
            if check_datasets_policy is not of allowed types
        """
        context = ModelComparisonContext(train_datasets, test_datasets, models)

        # Create progress bar
        progress_bar = ProgressBar(self.name, len(self.checks))

        # Run all checks
        results = []
        for check in self.checks.values():
            try:
                check_result = check.run_logic(context)
                results.append(check_result)
            except Exception as exp:
                results.append(CheckFailure(check, exp))
            progress_bar.inc_progress()

        progress_bar.close()
        return SuiteResult(self.name, results)


class ModelComparisonContext:
    """Contain processed input for model comparison checks."""

    def __init__(
        self,
        train_datasets: Union[Dataset, List[Dataset]],
        test_datasets: Union[Dataset, List[Dataset]],
        models: Union[List[Any], Mapping[str, Any]]
    ):
        """Preprocess the parameters."""
        # Validations
        if isinstance(train_datasets, Dataset) and isinstance(test_datasets, List):
            raise DeepchecksNotSupportedError('Single train dataset with multiple test datasets is not supported.')

        if not isinstance(models, (List, Mapping)):
            raise DeepchecksValueError('`models` must be a list or dictionary for compare models checks.')
        if len(models) < 2:
            raise DeepchecksValueError('`models` must receive 2 or more models')
        # Some logic to assign names to models
        if isinstance(models, List):
            models_dict = {}
            for m in models:
                model_type = type(m).__name__
                numerator = 1
                name = model_type
                while name in models_dict:
                    name = f'{model_type}_{numerator}'
                    numerator += 1
                models_dict[name] = m
            models = models_dict

        if not isinstance(train_datasets, List):
            train_datasets = [train_datasets] * len(models)
        if not isinstance(test_datasets, List):
            test_datasets = [test_datasets] * len(models)

        if len(train_datasets) != len(models):
            raise DeepchecksValueError('number of train_datasets must equal to number of models')
        if len(test_datasets) != len(models):
            raise DeepchecksValueError('number of test_datasets must equal to number of models')

        # Additional validations
        self.task_type = None
        self.contexts = []
        for i in range(len(models)):
            train = train_datasets[i]
            test = test_datasets[i]
            model = list(models.values())[i]
            name = list(models.keys())[i]
            context = Context(train, test, model, model_name=name)
            if self.task_type is None:
                self.task_type = context.task_type
            elif self.task_type != context.task_type:
                raise DeepchecksNotSupportedError('Got models of different task types')
            self.contexts.append(context)

    def __len__(self):
        """Return number of contexts."""
        return len(self.contexts)

    def __iter__(self):
        """Return iterator over context objects."""
        return iter(self.contexts)

    def __getitem__(self, item):
        """Return given context by index."""
        return self.contexts[item]


class ModelComparisonCheck(BaseCheck):
    """Parent class for check that compares between two or more models."""

    def __init__(self):
        """Initialize the class."""
        super().__init__()
        # Replace the run_logic function with wrapped run function
        setattr(self, 'run_logic', wrap_run(getattr(self, 'run_logic'), self))

    def run(self,
            train_datasets: Union[Dataset, List[Dataset]],
            test_datasets: Union[Dataset, List[Dataset]],
            models: Union[List[Any], Mapping[str, Any]]
            ) -> CheckResult:
        """Initialize context and pass to check logic."""
        return self.run_logic(ModelComparisonContext(train_datasets, test_datasets, models))

    @abc.abstractmethod
    def run_logic(self, multi_context: ModelComparisonContext) -> CheckResult:
        """Implement here logic of check."""
        raise NotImplementedError()
