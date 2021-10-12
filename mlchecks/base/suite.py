from collections import defaultdict
from typing import List, Dict, Union

from mlchecks.base.check import BaseCheck, CheckResult, TrainValidationBaseCheck, CompareDatasetsBaseCheck, \
    SingleDatasetBaseCheck, ModelOnlyBaseCheck, Decidable

__all__ = ['CheckSuite']

from mlchecks.utils import MLChecksValueError


class CheckSuite(BaseCheck):
    checks: List[BaseCheck]
    name: str

    def __init__(self, name, *checks, **params):
        super().__init__()
        self.name = name
        for check in checks:
            if not isinstance(check, BaseCheck):
                raise Exception(f'CheckSuite receives only `BaseCheck` objects but got: {check.__class__.__name__}')
        self.checks = checks

    def _run(self, model, train_dataset, validation_dataset, check_datasets_policy, decide=False) \
            -> Dict[str, Union[Dict, List[Union[CheckResult, bool]]]]:
        """
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

        results: Dict[str, Union[Dict, List[Union[CheckResult, bool]]]] = defaultdict(list)
        for check in self.checks:
            check_results = []
            if isinstance(check, TrainValidationBaseCheck):
                check_results.append(check.run(train_dataset=train_dataset, validation_dataset=validation_dataset,
                                                    model=model))
            elif isinstance(check, CompareDatasetsBaseCheck):
                check_results.append(check.run(dataset=train_dataset, compared_dataset=validation_dataset,
                                                    model=model))
            elif isinstance(check, SingleDatasetBaseCheck):
                if check_datasets_policy in ['both', 'train']:
                    check_results.append(check.run(dataset=train_dataset))
                if check_datasets_policy in ['both', 'validation']:
                    check_results.append(check.run(dataset=validation_dataset))
            elif isinstance(check, ModelOnlyBaseCheck):
                check_results.append(check.run(model=model))
            elif isinstance(check, CheckSuite):
                suite_res = check._run(model, train_dataset, validation_dataset, check_datasets_policy, decide)
                if check.name in results:
                    raise MLChecksValueError("Each suite must have a unique name")
                results[self.name].append(suite_res)
            else:
                raise TypeError(f'Expected check of type SingleDatasetBaseCheck, CompareDatasetsBaseCheck, '
                                f'TrainValidationBaseCheck or ModelOnlyBaseCheck. Got  {check.__class__.__name__} '
                                f'instead')

            if len(check_results) > 0:
                if not decide:
                    results[self.name].extend(check_results)
                else:
                    for res in check_results:
                        if isinstance(check, Decidable):
                            check: Decidable
                            results[self.name].append(check.decide(res))

        return results

    def run(self, model=None, train_dataset=None, validation_dataset=None, check_datasets_policy: str = 'validation') \
            -> Dict[str, Union[Dict, List[Union[CheckResult, bool]]]]:
        """
        Running the suites without validating the deciders.
        """
        return self._run(model, train_dataset, validation_dataset, check_datasets_policy, decide=False)

    def run_and_decide(self, model=None, train_dataset=None, validation_dataset=None,
                       check_datasets_policy: str = 'validation') \
            -> Dict[str, Union[Dict, List[Union[CheckResult, bool]]]]:

        return self._run(model, train_dataset, validation_dataset, check_datasets_policy, decide=True)




# [dataset_info, model_info, Suite("itay", mode_info)]
# {
#     "aaa" [sgdfg],
#
# }