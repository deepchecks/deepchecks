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
    DateTrainTestLeakageDuplicates().add_condition_leakage_ratio_not_greater_than(),
    DateTrainTestLeakageOverlap().add_condition_leakage_ratio_not_greater_than()
)

DataLeakageCheckSuite = CheckSuite(
    'Data Leakage Suite',
    DataSampleLeakageReport().add_condition_duplicates_ratio_not_greater_than(),
    # TODO: what default value to use for the condition
    SingleFeatureContribution().add_condition_feature_pps_not_greater_than(0.9),
    # TODO: what default value to use for the condition
    SingleFeatureContributionTrainTest().add_condition_feature_pps_difference_not_greater_than(0.1)
)

LeakageCheckSuite = CheckSuite(
    'Leakage Check Suite',
    IndexLeakageCheckSuite,
    DateLeakageCheckSuite,
    DataLeakageCheckSuite
)
