"""Module containing the Suite object, used for running a set of checks together."""
from typing import List

from IPython.core.display import display_html

from mlchecks.base.check import BaseCheck, CheckResult, TrainValidationBaseCheck, CompareDatasetsBaseCheck, \
    SingleDatasetBaseCheck, ModelOnlyBaseCheck

__all__ = ['CheckSuite']

from mlchecks.utils import MLChecksValueError


class CheckSuite(BaseCheck):
    """Class for running a set of checks together, and returning a unified pass / no-pass.

    Attributes:
        checks: A list of checks to run.
    """

    checks: List[BaseCheck]
    name: str

    def __init__(self, name, *checks):
        """Get `Check`s and `CheckSuite`s to run in given order."""
        super().__init__()
        for check in checks:
            if not isinstance(check, BaseCheck):
                raise Exception(f'CheckSuite receives only `BaseCheck` objects but got: {check.__class__.__name__}')
        self.checks = checks
        self.name = name

    def run(self, model=None, train_dataset=None, validation_dataset=None, check_datasets_policy: str = 'validation') \
            -> List[CheckResult]:
        """Run all checks.

        Args:
          model: A scikit-learn-compatible fitted estimator instance
          train_dataset: Dataset object, representing data an estimator was fitted on
          validation_dataset: Dataset object, representing data an estimator predicts on
          check_datasets_policy: str, one of either ['both', 'train', 'validation'].
                                 Determines the policy by which single dataset checks are run when two datasets are
                                 given, one for train and the other for validation.

        Returns:
          List[CheckResult] - All results by all initialized checks

        Raises:
             ValueError if check_datasets_policy is not of allowed types
        """
        if check_datasets_policy not in ['both', 'train', 'validation']:
            raise ValueError('check_datasets_policy must be one of ["both", "train", "validation"]')

        results = []
        for check in self.checks:
            if isinstance(check, TrainValidationBaseCheck):
                results.append(check.run(train_dataset=train_dataset, validation_dataset=validation_dataset,
                                         model=model))
            elif isinstance(check, CompareDatasetsBaseCheck):
                results.append(check.run(dataset=train_dataset, compared_dataset=validation_dataset, model=model))
            elif isinstance(check, SingleDatasetBaseCheck):
                if check_datasets_policy in ['both', 'train'] and train_dataset is not None:
                    results.append(check.run(dataset=train_dataset))
                if check_datasets_policy in ['both', 'validation'] and validation_dataset is not None:
                    results.append(check.run(dataset=validation_dataset))
            elif isinstance(check, ModelOnlyBaseCheck):
                results.append(check.run(model=model))
            elif isinstance(check, CheckSuite):
                suite_res = check.run(model, train_dataset, validation_dataset, check_datasets_policy)
                if check.name in results:
                    raise MLChecksValueError('Each suite must have a unique name')
                results.append(suite_res)
            else:
                raise TypeError(f'Expected check of type SingleDatasetBaseCheck, CompareDatasetsBaseCheck, '
                                f'TrainValidationBaseCheck or ModelOnlyBaseCheck. Got  {check.__class__.__name__} '
                                f'instead')

        def display():
            display_html(f'<h3>{self.name}</h3>', raw=True)
            for result in results:
                result._ipython_display_()

        return CheckResult(results, display=display)
