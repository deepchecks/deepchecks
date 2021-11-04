"""The predefined Leakage suite module."""
from mlchecks import CheckSuite
from mlchecks.checks.leakage import (
    DataSampleLeakageReport,
    DateTrainValidationLeakageDuplicates,
    DateTrainValidationLeakageOverlap,
    IndexTrainValidationLeakage,
    SingleFeatureContributionTrainValidation
)


__all__ = ['IndexLeakageCheckSuite', 'DateLeakageCheckSuite', 'DataLeakageCheckSuite', 'LeakageCheckSuite']

IndexLeakageCheckSuite = CheckSuite(
    'Index Leakage Suite',
    DateTrainValidationLeakageDuplicates(),
    IndexTrainValidationLeakage(),
)

DateLeakageCheckSuite = CheckSuite(
    'Date Leakage Suite',
    DateTrainValidationLeakageDuplicates(),
    DateTrainValidationLeakageOverlap(),
)

DataLeakageCheckSuite = CheckSuite(
    'Data Leakage Suite',
    DataSampleLeakageReport(),
    SingleFeatureContributionTrainValidation()
)

LeakageCheckSuite = CheckSuite(
    'Leakage Check Suite',
    IndexLeakageCheckSuite(),
    DateLeakageCheckSuite(),
    DataLeakageCheckSuite()
)

