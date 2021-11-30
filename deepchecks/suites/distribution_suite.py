"""The predefined Data Distribution suite module."""
from deepchecks import Suite
from deepchecks.checks.distribution import TrainTestDrift
from deepchecks.checks.distribution import TrustScoreComparison


def data_distribution_suite() -> Suite:
    """Create 'Data Distribution Suite'.

    The suite runs a check comparing the distributions of the training and test datasets.
    """
    return Suite(
        'Data Distribution',
        TrainTestDrift().add_condition_drift_score_not_greater_than(),
        TrustScoreComparison().add_condition_mean_score_percent_decline_not_greater_than()
    )
