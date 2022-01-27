# ----------------------------------------------------------------------------
# Copyright (C) 2021 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""Module for base tabular abstractions."""
# TODO: This file should be completely modified
# pylint: disable=broad-except
import abc
from collections import OrderedDict
from typing import Tuple, Mapping, Optional

from ignite.metrics import Metric
from torch import nn

from deepchecks.vision.utils.validation import model_type_validation
from deepchecks.vision.utils.metrics import TaskType, task_type_check
from deepchecks.core.check import CheckResult, BaseCheck, CheckFailure, wrap_run
from deepchecks.core.suite import BaseSuite, SuiteResult
from deepchecks.core.display_suite import ProgressBar
from deepchecks.core.errors import (
    DatasetValidationError, ModelValidationError,
    DeepchecksNotSupportedError, DeepchecksValueError
)
from deepchecks.vision import VisionDataset


__all__ = [
    'Context',
    'Suite',
    'Check',
    'SingleDatasetBaseCheck',
    'TrainTestBaseCheck',
    'ModelOnlyBaseCheck',
]


class Context:
    """Contains all the data + properties the user has passed to a check/suite, and validates it seamlessly.

    Parameters
    ----------
    train : VisionDataset , default: None
        Dataset or DataFrame object, representing data an estimator was fitted on
    test : VisionDataset , default: None
        Dataset or DataFrame object, representing data an estimator predicts on
    model : BasicModel , default: None
        A scikit-learn-compatible fitted estimator instance
    model_name: str , default: ''
        The name of the model
    scorers : Mapping[str, Metric] , default: None
        dict of scorers names to a Metric
    scorers_per_class : Mapping[str, Metric] , default: None
        dict of scorers for classification without averaging of the classes.
        See <a href=
        "https://scikit-learn.org/stable/modules/model_evaluation.html#from-binary-to-multiclass-and-multilabel">
        scikit-learn docs</a>
    """

    def __init__(self,
                 train: VisionDataset = None,
                 test: VisionDataset = None,
                 model: nn.Module = None,
                 model_name: str = '',
                 scorers: Mapping[str, Metric] = None,
                 scorers_per_class: Mapping[str, Metric] = None
                 ):
        # Validations
        if train is None and test is None and model is None:
            raise DeepchecksValueError('At least one dataset (or model) must be passed to the method!')
        # if train is not None:
        #     train = VisionDataset.ensure_not_empty_dataset(train)
        # if test is not None:
        #     test = VisionDataset.ensure_not_empty_dataset(test)
        # # If both dataset, validate they fit each other
        # if train and test:
        #     if not VisionDataset.datasets_share_label(train, test):
        #         raise DatasetValidationError('train and test requires to have and to share the same label')
        #     if not VisionDataset.datasets_share_features(train, test):
        #         raise DatasetValidationError('train and test requires to share the same features columns')
        #     if not VisionDataset.datasets_share_categorical_features(train, test):
        #         raise DatasetValidationError(
        #             'train and test datasets should share '
        #             'the same categorical features. Possible reason is that some columns were'
        #             'inferred incorrectly as categorical features. To fix this, manually edit the '
        #             'categorical features using Dataset(cat_features=<list_of_features>'
        #         )
        #     if not VisionDataset.datasets_share_index(train, test):
        #         raise DatasetValidationError('train and test requires to share the same index column')
        #     if not VisionDataset.datasets_share_date(train, test):
        #         raise DatasetValidationError('train and test requires to share the same date column')
        if test and not train:
            raise DatasetValidationError('Can\'t initialize context with only test. if you have single dataset, '
                                         'initialize it as train')
        if model is not None:
            # Here validate only type of model, later validating it can predict on the data if needed
            model_type_validation(model)

        self._train = train
        self._test = test
        self._model = model
        self._validated_model = False
        self._task_type = None
        self._user_scorers = scorers
        self._user_scorers_per_class = scorers_per_class
        self._model_name = model_name

    # Properties
    # Validations note: We know train & test fit each other so all validations can be run only on train

    @property
    def train(self) -> VisionDataset:
        """Return train if exists, otherwise raise error."""
        if self._train is None:
            raise DeepchecksNotSupportedError('Check is irrelevant for Datasets without train dataset')
        return self._train

    @property
    def test(self) -> VisionDataset:
        """Return test if exists, otherwise raise error."""
        if self._test is None:
            raise DeepchecksNotSupportedError('Check is irrelevant for Datasets without test dataset')
        return self._test

    @property
    def model(self) -> nn.Module:
        """Return & validate model if model exists, otherwise raise error."""
        if self._model is None:
            raise DeepchecksNotSupportedError('Check is irrelevant for Datasets without model')
        if not self._validated_model:
            # if self._train:
            #     validate_model(self._train, self._model)
            self._validated_model = True
        return self._model

    @property
    def model_name(self):
        """Return model name."""
        return self._model_name

    @property
    def task_type(self) -> TaskType:
        """Return task type if model & train & label exists. otherwise, raise error."""
        if self._task_type is None:
            self._task_type = task_type_check(self.model, self.train)
        return self._task_type

    def have_test(self):
        """Return whether there is test dataset defined."""
        return self._test is not None

    def assert_task_type(self, *expected_types: TaskType):
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
        if (not self.assert_task_type(TaskType.CLASSIFICATION) and
                self.train.label_type == 'regression_label'):
            raise ModelValidationError('Check is irrelevant for regressions tasks')

    def assert_regression_task(self):
        """Assert the task type is regression."""
        # assert_task_type makes assertion if task type exists and returns True, else returns False
        # If not task type than check label type
        if (not self.assert_task_type(TaskType.REGRESSION) and
                self.train.label_type == 'classification_label'):
            raise ModelValidationError('Check is irrelevant for classification tasks')

    # def get_scorers(self, alternative_scorers: Mapping[str, Union[str, Callable]] = None, class_avg=True):
    #     """Return initialized & validated scorers in a given priority.
    #
    #     If receive `alternative_scorers` return them,
    #     Else if user defined global scorers return them,
    #     Else return default scorers.
    #
    #     Parameters
    #     ----------
    #     alternative_scorers : Mapping[str, Union[str, Callable]], default None
    #         dict of scorers names to scorer sklearn_name/function
    #     class_avg : bool, default True
    #         for classification whether to return scorers of average score or score per class
    #     """
    #     if class_avg:
    #         user_scorers = self._user_scorers
    #     else:
    #         user_scorers = self._user_scorers_per_class
    #
    #     scorers = alternative_scorers or user_scorers or get_default_scorers(self.task_type, class_avg)
    #     return init_validate_scorers(scorers, self.model, self.train, class_avg, self.task_type)
    #
    # def get_single_scorer(self, alternative_scorers: Mapping[str, Union[str, Callable]] = None, class_avg=True):
    #     """Return initialized & validated single scorer in a given priority.
    #
    #     If receive `alternative_scorers` use them,
    #     Else if user defined global scorers use them,
    #     Else use default scorers.
    #     Returns the first scorer from the scorers described above.
    #
    #     Parameters
    #     ----------
    #     alternative_scorers : Mapping[str, Union[str, Callable]], default None
    #         dict of scorers names to scorer sklearn_name/function. Only first scorer will be used.
    #     class_avg : bool, default True
    #         for classification whether to return scorers of average score or score per class
    #     """
    #     if class_avg:
    #         user_scorers = self._user_scorers
    #     else:
    #         user_scorers = self._user_scorers_per_class
    #
    #     scorers = alternative_scorers or user_scorers or get_default_scorers(self.task_type, class_avg)
    #     # The single scorer is the first one in the dict
    #     scorer_name = next(iter(scorers))
    #     single_scorer_dict = {scorer_name: scorers[scorer_name]}
    #     return init_validate_scorers(single_scorer_dict, self.model, self.train, class_avg, self.task_type)[0]


class Check(BaseCheck):
    """Base class for all tabular checks."""

    def __init__(self):
        # pylint: disable=super-init-not-called
        self._conditions = OrderedDict()
        self._conditions_index = 0
        # Replace the run_logic function with wrapped run function
        setattr(self, 'run_logic', wrap_run(getattr(self, 'run_logic'), self))

    def run_logic(self, context: Context, **kwargs) -> CheckResult:
        """Run check logic."""
        raise NotImplementedError()


class SingleDatasetBaseCheck(Check):
    """Parent class for checks that only use one dataset."""

    def run(self, dataset, model=None) -> CheckResult:
        """Run check."""
        # By default, we initialize a single dataset as the "train"
        c = Context(dataset, model=model)
        return self.run_logic(c)

    @abc.abstractmethod
    def run_logic(self, context: Context, dataset_type: str = 'train') -> CheckResult:
        """Run check."""
        pass


class TrainTestBaseCheck(Check):
    """Parent class for checks that compare two datasets.

    The class checks train dataset and test dataset for model training and test.
    """

    def run(self, train_dataset, test_dataset, model=None) -> CheckResult:
        """Run check."""
        c = Context(train_dataset, test_dataset, model=model)
        return self.run_logic(c)


class ModelOnlyBaseCheck(Check):
    """Parent class for checks that only use a model and no datasets."""

    def run(self, model) -> CheckResult:
        """Run check."""
        return self.run_logic(Context(model=model))


class Suite(BaseSuite):
    """Tabular suite to run checks of types: TrainTestBaseCheck, SingleDatasetBaseCheck, ModelOnlyBaseCheck."""

    @classmethod
    def supported_checks(cls) -> Tuple:
        """Return tuple of supported check types of this suite."""
        return TrainTestBaseCheck, SingleDatasetBaseCheck, ModelOnlyBaseCheck

    def run(
            self,
            train_dataset: Optional[VisionDataset] = None,
            test_dataset: Optional[VisionDataset] = None,
            model: nn.Module = None,
            scorers: Mapping[str, Metric] = None,
            scorers_per_class: Mapping[str, Metric] = None
    ) -> SuiteResult:
        """Run all checks.

        Parameters
        ----------
        train_dataset: Optional[VisionDataset] , default None
            object, representing data an estimator was fitted on
        test_dataset : Optional[VisionDataset] , default None
            object, representing data an estimator predicts on
        model : nn.Module , default None
            A scikit-learn-compatible fitted estimator instance
        scorers : Mapping[str, Metric] , default None
            dict of scorers names to scorer sklearn_name/function
        scorers_per_class : Mapping[str, Metric], default None
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
                          scorers=scorers,
                          scorers_per_class=scorers_per_class)
        # Create progress bar
        progress_bar = ProgressBar(self.name, len(self.checks))

        # Run all checks
        results = []
        for check in self.checks.values():
            try:
                progress_bar.set_text(check.name())
                if isinstance(check, TrainTestBaseCheck):
                    if train_dataset is not None and test_dataset is not None:
                        check_result = check.run_logic(context)
                        results.append(check_result)
                    else:
                        msg = 'Check is irrelevant if not supplied with both train and test datasets'
                        results.append(Suite._get_unsupported_failure(check, msg))
                elif isinstance(check, SingleDatasetBaseCheck):
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
                elif isinstance(check, ModelOnlyBaseCheck):
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

