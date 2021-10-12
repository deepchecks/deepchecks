"""The single_feature_contribution check module."""
from typing import Union
import pandas as pd
import ppscore as pps
import seaborn as sns

from mlchecks import CheckResult, Dataset, SingleDatasetBaseCheck
from mlchecks.utils import is_notebook, MLChecksValueError, get_plt_html_str

__all__ = ['single_feature_contribution', 'SingleFeatureContribution']


def single_feature_contribution(dataset: Union[Dataset, pd.DataFrame]):
    """
    Return the PPS (Predictive Power Score) of all features in relation to the label.
    The PPS represents the ability of a feature to single-handedly predict another feature or label.
    A high PPS (close to 1) can mean that this feature's success in predicting the label is actually due to data
    leakage - meaning that the feature holds information that is based on the label to begin with.

   Args:
       dataset (Dataset): A dataset object. Must contain a label.
   Returns:
       CheckResult:
            - value is the highest PPS a feature has got.
            - data is a bar graph of the PPS of each feature.

    Raises:
        MLChecksValueError: If the object is not a Dataset instance with a label

    """

    if not isinstance(dataset, pd.DataFrame):
        raise MLChecksValueError("single_feature_contribution check must receive a Dataset object")
    if not dataset.label():
        raise MLChecksValueError("single_feature_contribution requires dataset to have a label.")

    relevant_columns = dataset.features + [dataset.label()]
    df_pps = pps.predictors(dataset[relevant_columns], dataset.label())
    max_pps = df_pps.ppscore.max()

    # Create graph:
    sns.barplot(data=df_pps, x="x", y="ppscore")
    html = get_plt_html_str()  # Catches graph into html

    return CheckResult(value=max_pps, display={'text/html': html})


class SingleFeatureContribution(SingleDatasetBaseCheck):
    """
    Return the PPS (Predictive Power Score) of all features in relation to the label.
    The PPS represents the ability of a feature to single-handedly predict another feature or label.
    A high PPS (close to 1) can mean that this feature's success in predicting the label is actually due to data
    leakage - meaning that the feature holds information that is based on the label to begin with.
    """

    def run(self, dataset: Dataset, model=None) -> CheckResult:
        """
        Runs the single_feature_contribution check

        Arguments:
            dataset: Dataset - The dataset object
            model: any = None - not used in the check

        Returns:
            the output of the dataset_info check
        """
        return single_feature_contribution(dataset)

