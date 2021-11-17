"""The predefined overfit suite module."""
from deepchecks.suites import IntegrityCheckSuite, DataLeakageCheckSuite, OverfitCheckSuite, PerformanceCheckSuite, \
    ClassificationCheckSuite, RegressionCheckSuite, GenericPerformanceCheckSuite, LeakageCheckSuite
from deepchecks import CheckSuite


__all__ = ['OverallCheckSuite', 'OverallClassificationCheckSuite', 'OverallRegressionCheckSuite',
           'OverallGenericCheckSuite']


OverallCheckSuite = CheckSuite(
    'Overall Suite',
    LeakageCheckSuite,
    OverfitCheckSuite,
    PerformanceCheckSuite,
    IntegrityCheckSuite,
)


OverallClassificationCheckSuite = CheckSuite(
    'Overall Classification Suite',
    DataLeakageCheckSuite,
    OverfitCheckSuite,
    ClassificationCheckSuite,
    IntegrityCheckSuite,
)


OverallRegressionCheckSuite = CheckSuite(
    'Overall Regression Suite',
    DataLeakageCheckSuite,
    OverfitCheckSuite,
    RegressionCheckSuite,
    IntegrityCheckSuite,
)


OverallGenericCheckSuite = CheckSuite(
    'Overall Generic Suite',
    DataLeakageCheckSuite,
    OverfitCheckSuite,
    GenericPerformanceCheckSuite,
    IntegrityCheckSuite,
)
