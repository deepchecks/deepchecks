"""The predefined overfit suite module."""
from mlchecks.suites import IntegrityCheckSuite, DataLeakageCheckSuite, OverfitCheckSuite, PerformanceCheckSuite, \
    ClassificationCheckSuite, RegressionCheckSuite, GenericPerformanceCheckSuite, LeakageCheckSuite
from mlchecks import CheckSuite


__all__ = ['OverallCheckSuite', 'OverallClassificationCheckSuite', 'OverallRegressionCheckSuite',
           'OverallGenericCheckSuite']


OverallCheckSuite = CheckSuite(
    'Overall Suite',
    IntegrityCheckSuite,
    LeakageCheckSuite,
    OverfitCheckSuite,
    PerformanceCheckSuite
)


OverallClassificationCheckSuite = CheckSuite(
    'Overall Classification Suite',
    IntegrityCheckSuite,
    DataLeakageCheckSuite,
    OverfitCheckSuite,
    ClassificationCheckSuite
)


OverallRegressionCheckSuite = CheckSuite(
    'Overall Regression Suite',
    IntegrityCheckSuite,
    DataLeakageCheckSuite,
    OverfitCheckSuite,
    RegressionCheckSuite
)


OverallGenericCheckSuite = CheckSuite(
    'Overall Generic Suite',
    IntegrityCheckSuite,
    DataLeakageCheckSuite,
    OverfitCheckSuite,
    GenericPerformanceCheckSuite
)