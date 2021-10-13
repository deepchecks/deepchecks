"""The single_feature_contribution check module."""
from typing import Union
import pandas as pd
import ppscore as pps
import seaborn as sns

from mlchecks import CheckResult, Dataset, SingleDatasetBaseCheck
from mlchecks.base.dataset import validate_dataset
from mlchecks.display import format_check_display
from mlchecks.utils import get_plt_html_str

__all__ = ['single_feature_contribution', 'SingleFeatureContribution']


def single_feature_contribution(dataset: Union[Dataset, pd.DataFrame], ppscore_params=None):
    """
    Return the PPS (Predictive Power Score) of all features in relation to the label.

    The PPS represents the ability of a feature to single-handedly predict another feature or label.
    A high PPS (close to 1) can mean that this feature's success in predicting the label is actually due to data
    leakage - meaning that the feature holds information that is based on the label to begin with.

    Uses the ppscore package - for more info, see https://github.com/8080labs/ppscore

    Args:
        dataset (Dataset): A dataset object. Must contain a label
        ppscore_params (dict): dictionary of addional paramaters for the ppscore.predictors function
    Returns:
        CheckResult:
            value is a dictionary with PPS per feature column.
            data is a bar graph of the PPS of each feature.

    Raises:
        MLChecksValueError: If the object is not a Dataset instance with a label

    """
    dataset = validate_dataset(dataset, 'single_feature_contribution')
    dataset.validate_label('single_feature_contribution')
    ppscore_params = ppscore_params or {}

    relevant_columns = dataset.features() + [dataset.label_name()]
    df_pps = pps.predictors(df=dataset[relevant_columns], y=dataset.label_name(), random_seed=42, **ppscore_params)
    df_pps = df_pps.set_index('x', drop=True)
    s_ppscore = df_pps['ppscore']

    # Create graph:
    sns.barplot(x=s_ppscore.index, y=s_ppscore)
    html = get_plt_html_str()  # Catches graph into html
    formatted_html = format_check_display('Single Feature Contribution', single_feature_contribution, html)

    return CheckResult(value=s_ppscore.to_dict(), display={'text/html': formatted_html})


class SingleFeatureContribution(SingleDatasetBaseCheck):
    """
    Return the PPS (Predictive Power Score) of all features in relation to the label.

    The PPS represents the ability of a feature to single-handedly predict another feature or label.
    A high PPS (close to 1) can mean that this feature's success in predicting the label is actually due to data
    leakage - meaning that the feature holds information that is based on the label to begin with.

    Uses the ppscore package - for more info, see https://github.com/8080labs/ppscore

    """

    def run(self, dataset: Dataset, model=None) -> CheckResult:
        """
        Run the single_feature_contribution check.

        Arguments:
            dataset: Dataset - The dataset object
            model: any = None - not used in the check

        Returns:
            the output of the dataset_info check
        """
        return single_feature_contribution(dataset=dataset, ppscore_params=self.params.get('ppscore_params'))

