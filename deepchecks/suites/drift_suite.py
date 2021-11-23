"""The predefined Drift suite module."""
from deepchecks import CheckSuite
from deepchecks.checks.drift import TrainTestDrift


def train_test_drift_suite() -> CheckSuite:
    """Create 'Train Test Drift Suite'.

    The suite runs a check comparing the distributions of the training and test datasets.
    """
    return CheckSuite(
        'Train Test Drift Suite',
        TrainTestDrift.add_condition_drift_score_not_greater_than()
    )
