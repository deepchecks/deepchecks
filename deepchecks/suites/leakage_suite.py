"""The predefined Leakage suite module."""
from deepchecks import CheckSuite
from deepchecks.checks.leakage import (
    DataSampleLeakageReport,
    DateTrainTestLeakageDuplicates,
    DateTrainTestLeakageOverlap,
    IndexTrainTestLeakage,
    SingleFeatureContributionTrainTest,
    SingleFeatureContribution
)


__all__ = ['IndexLeakageCheckSuite', 'DateLeakageCheckSuite', 'DataLeakageCheckSuite', 'LeakageCheckSuite']

IndexLeakageCheckSuite = CheckSuite(
    'Index Leakage Suite',
    IndexTrainTestLeakage(),
)

DateLeakageCheckSuite = CheckSuite(
    'Date Leakage Suite',
    DateTrainTestLeakageDuplicates(),
    DateTrainTestLeakageOverlap()
)

DataLeakageCheckSuite = CheckSuite(
    'Data Leakage Suite',
    DataSampleLeakageReport(),
    SingleFeatureContribution(),
    SingleFeatureContributionTrainTest()
)

LeakageCheckSuite = CheckSuite(
    'Leakage Check Suite',
    IndexLeakageCheckSuite,
    DateLeakageCheckSuite,
    DataLeakageCheckSuite
)

