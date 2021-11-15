"""The predefined Leakage suite module."""
from deepchecks import CheckSuite
from deepchecks.checks.leakage import (
    DataSampleLeakageReport,
    DateTrainValidationLeakageDuplicates,
    DateTrainValidationLeakageOverlap,
    IndexTrainValidationLeakage,
    SingleFeatureContributionTrainValidation,
    SingleFeatureContribution
)


__all__ = ['IndexLeakageCheckSuite', 'DateLeakageCheckSuite', 'DataLeakageCheckSuite', 'LeakageCheckSuite']

IndexLeakageCheckSuite = CheckSuite(
    'Index Leakage Suite',
    IndexTrainValidationLeakage(),
)

DateLeakageCheckSuite = CheckSuite(
    'Date Leakage Suite',
    DateTrainValidationLeakageDuplicates(),
    DateTrainValidationLeakageOverlap()
)

DataLeakageCheckSuite = CheckSuite(
    'Data Leakage Suite',
    DataSampleLeakageReport(),
    SingleFeatureContribution(),
    SingleFeatureContributionTrainValidation()
)

LeakageCheckSuite = CheckSuite(
    'Leakage Check Suite',
    IndexLeakageCheckSuite,
    DateLeakageCheckSuite,
    DataLeakageCheckSuite
)

