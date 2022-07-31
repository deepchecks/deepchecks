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
"""Module for base tabular model abstractions."""
# pylint: disable=broad-except
from typing import Any, Dict, List, Mapping, Tuple, Union

from deepchecks.core.check_result import CheckFailure, CheckResult
from deepchecks.core.errors import DeepchecksNotSupportedError, DeepchecksValueError
from deepchecks.core.suite import BaseSuite, SuiteResult
from deepchecks.tabular.context import Context
from deepchecks.tabular.dataset import Dataset
from deepchecks.utils.ipython import create_progress_bar

__all__ = [
    'ModelComparisonSuite',
    'ModelComparisonContext'
]


class ModelComparisonSuite(BaseSuite):
    """Suite to run checks of types: CompareModelsBaseCheck."""

    @classmethod
    def supported_checks(cls) -> Tuple:
        """Return tuple of supported check types of this suite."""
        from deepchecks.tabular.base_checks import ModelComparisonCheck  # pylint: disable=import-outside-toplevel
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
        progress_bar = create_progress_bar(
            iterable=list(self.checks.values()),
            name=self.name,
            unit='Check'
        )

        # Run all checks
        results = []

        for check in progress_bar:
            try:
                check_result = check.run_logic(context)
                results.append(check_result)
            except Exception as exp:
                results.append(CheckFailure(check, exp))

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
            context = Context(train, test, model)
            if self.task_type is None:
                self.task_type = context.task_type
            elif self.task_type != context.task_type:
                raise DeepchecksNotSupportedError('Got models of different task types')
            self.contexts.append(context)
        self._models = models

    @property
    def models(self) -> Dict:
        """Return the models' dict."""
        return self._models

    def __len__(self):
        """Return number of contexts."""
        return len(self.contexts)

    def __iter__(self):
        """Return iterator over context objects."""
        return iter(self.contexts)

    def __getitem__(self, item):
        """Return given context by index."""
        return self.contexts[item]

    def finalize_check_result(self, check_result, check):
        """Run final processing on a check result which includes validation and conditions processing."""
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
