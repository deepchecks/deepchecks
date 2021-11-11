"""The single_feature_contribution check module."""
import mlchecks.ppscore as pps
from mlchecks import CheckResult, Dataset, SingleDatasetBaseCheck
from mlchecks.plot_utils import create_colorbar_barchart_for_check

__all__ = ['SingleFeatureContribution']


class SingleFeatureContribution(SingleDatasetBaseCheck):
    """Return the PPS (Predictive Power Score) of all features in relation to the label.

    The PPS represents the ability of a feature to single-handedly predict another feature or label.
    A high PPS (close to 1) can mean that this feature's success in predicting the label is actually due to data
    leakage - meaning that the feature holds information that is based on the label to begin with.

    Uses the ppscore package - for more info, see https://github.com/8080labs/ppscore

    """

    def __init__(self, ppscore_params=None, n_show_top: int = 5):
        """Initialize the SingleFeatureContribution check.

        Args:
            ppscore_params (dict): dictionary of addional paramaters for the ppscore.predictors function
        """
        super().__init__()
        self.ppscore_params = ppscore_params
        self.n_show_top = n_show_top

    def run(self, dataset: Dataset, model=None) -> CheckResult:
        """Run check.

        Arguments:
            dataset: Dataset - The dataset object
            model: any = None - not used in the check

        Returns:
            CheckResult:
                value is a dictionary with PPS per feature column.
                data is a bar graph of the PPS of each feature.

        Raises:
            MLChecksValueError: If the object is not a Dataset instance with a label
        """
        return self._single_feature_contribution(dataset=dataset)

    def _single_feature_contribution(self, dataset: Dataset):
        Dataset.validate_dataset(dataset, self.__class__.__name__)
        dataset.validate_label(self.__class__.__name__)
        ppscore_params = self.ppscore_params or {}

        relevant_columns = dataset.features() + [dataset.label_name()]
        df_pps = pps.predictors(df=dataset.data[relevant_columns], y=dataset.label_name(), random_seed=42,
                                **ppscore_params)
        df_pps = df_pps.set_index('x', drop=True).head(self.n_show_top)
        s_ppscore = df_pps['ppscore']

        def plot():
            # Create graph:
            create_colorbar_barchart_for_check(x=s_ppscore.index, y=s_ppscore.values,
                                               check_name=self._single_feature_contribution.__name__)

        text = ['The PPS represents the ability of a feature to single-handedly predict another feature or label.',
                'A high PPS (close to 1) can mean that this feature\'s success in predicting the label is'
                ' actually due to data',
                'leakage - meaning that the feature holds information that is based on the label to begin with.']

        return CheckResult(value=s_ppscore.to_dict(), display=[plot, *text], check=self.__class__,
                           header='Single Feature Contribution')
