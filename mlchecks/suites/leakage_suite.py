"""The predefined Leakage suite module."""
from mlchecks import CheckSuite
from mlchecks.checks.leakage import (
    DataSampleLeakageReport,
    DateTrainValidationLeakageDuplicates,
    DateTrainValidationLeakageOverlap,
    IndexTrainValidationLeakage,
    SingleFeatureContributionTrainValidation
)


__all__ = ['LeakageCheckSuite']


LeakageCheckSuite = CheckSuite(
    'Leakage Suite',
    DataSampleLeakageReport(),
    DateTrainValidationLeakageDuplicates(),
    DateTrainValidationLeakageOverlap(),
    IndexTrainValidationLeakage(),
    SingleFeatureContributionTrainValidation()
)
