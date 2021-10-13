"""The single_feature_contribution check module."""
from typing import Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import ppscore as pps

from mlchecks import CheckResult, Dataset, SingleDatasetBaseCheck, TrainValidationBaseCheck
from mlchecks.base.dataset import validate_dataset
from mlchecks.utils import get_plt_html_str, get_txt_html_str

__all__ = ['single_feature_contribution', 'single_feature_contribution_train_validation',
           'SingleFeatureContribution', 'SingleFeatureContributionTrainValidation']


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
    ppscore_params = ppscore_params or dict()

    relevant_columns = dataset.features() + [dataset.label_name()]
    df_pps = pps.predictors(df=dataset[relevant_columns], y=dataset.label_name(), random_seed=42, **ppscore_params)
    df_pps = df_pps.set_index('x', drop=True)
    s_ppscore = df_pps['ppscore']

    # Create graph:
    # s_ppscore.plot(kind='bar', ylabel='ppscore', ylim=(0, 1), grid=True)
    create_colorbar_barchart_for_check(x=s_ppscore.index, y=s_ppscore.values)

    html_plot = get_plt_html_str()  # Catches graph into html
    html_txt = get_txt_html_str(
        ['The PPS represents the ability of a feature to single-handedly predict another feature or label.',
         'A high PPS (close to 1) can mean that this feature\'s success in predicting the label is actually due to data',
         'leakage - meaning that the feature holds information that is based on the label to begin with.'])

    return CheckResult(value=s_ppscore.to_dict(), display={'text/html': html_txt + html_plot})


def single_feature_contribution_train_validation(train_dataset: Dataset, validation_dataset: Dataset,
                                                 ppscore_params=None):
    """
    Return the difference in PPS (Predictive Power Score) of all features between train and validation datasets.

    The PPS represents the ability of a feature to single-handedly predict another feature or label.
    A high PPS (close to 1) can mean that this feature's success in predicting the label is actually due to data
    leakage - meaning that the feature holds information that is based on the label to begin with.
    A high difference in PPS between train and validation can indicate leakage as well.

    Uses the ppscore package - for more info, see https://github.com/8080labs/ppscore

    Args:
        train_dataset (Dataset): A dataset object. Must contain a label
        validation_dataset (Dataset): A dataset object. Must contain a label
        ppscore_params (dict): dictionary of addional paramaters for the ppscore.predictors function

    Returns:
        CheckResult:
            value is a dictionary with PPS per feature column.
            data is a bar graph of the PPS of each feature.

    Raises:
        MLChecksValueError: If the object is not a Dataset instance with a label

    """
    func_name = 'single_feature_contribution'
    train_dataset = validate_dataset(train_dataset, func_name)
    train_dataset.validate_label(func_name)
    validation_dataset = validate_dataset(validation_dataset, func_name)
    validation_dataset.validate_label(func_name)
    features_names = train_dataset.validate_shared_features(validation_dataset, func_name)
    label_name = train_dataset.validate_shared_label(validation_dataset, func_name)
    ppscore_params = ppscore_params or dict()

    relevant_columns = features_names + [label_name]
    df_pps_train = pps.predictors(df=train_dataset[relevant_columns], y=train_dataset.label_name(), random_seed=42,
                                  **ppscore_params)
    df_pps_validation = pps.predictors(df=validation_dataset[relevant_columns], y=validation_dataset.label_name(),
                                       random_seed=42, **ppscore_params)
    s_pps_train = df_pps_train.set_index('x', drop=True)['ppscore']
    s_pps_validation = df_pps_validation.set_index('x', drop=True)['ppscore']

    s_difference = s_pps_train - s_pps_validation
    s_difference = s_difference.apply(lambda x: 0 if x < 0 else x)

    # Create graph:
    create_colorbar_barchart_for_check(x=s_difference.index, y=s_difference.values)

    html_plot = get_plt_html_str()  # Catches graph into html
    html_txt = get_txt_html_str(
        ['The PPS represents the ability of a feature to single-handedly predict another feature or label.',
         'A high PPS (close to 1) can mean that this feature\'s success in predicting the label is actually due to data',
         'leakage - meaning that the feature holds information that is based on the label to begin with.',
         'A high difference in PPS between train and validation can indicate leakage as well.'])

    return CheckResult(value=s_difference.to_dict(), display={'text/html': html_txt + html_plot})


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


class SingleFeatureContributionTrainValidation(TrainValidationBaseCheck):
    """
    Return the difference in PPS (Predictive Power Score) of all features between train and validation datasets.

    The PPS represents the ability of a feature to single-handedly predict another feature or label.
    A high PPS (close to 1) can mean that this feature's success in predicting the label is actually due to data
    leakage - meaning that the feature holds information that is based on the label to begin with.

    Uses the ppscore package - for more info, see https://github.com/8080labs/ppscore

    """

    def run(self, train_dataset: Dataset, validation_dataset: Dataset, model=None) -> CheckResult:
        """
        Run the single_feature_contribution check.

        Arguments:
            train_dataset (Dataset): A dataset object. Must contain a label
            validation_dataset (Dataset): A dataset object. Must contain a label
            model: any = None - not used in the check

        Returns:
            the output of the dataset_info check
        """
        return single_feature_contribution_train_validation(train_dataset=train_dataset,
                                                            validation_dataset=validation_dataset,
                                                            ppscore_params=self.params.get('ppscore_params'))


# Utils:


def create_colorbar_barchart_for_check(x: np.array, y: np.array):
    fig, ax = plt.subplots(figsize=(15, 4))

    my_cmap = plt.cm.get_cmap('RdYlGn_r')
    colors = my_cmap([h for h in y])
    rects = ax.bar(x, y, color=colors)

    sm = ScalarMappable(cmap=my_cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])

    cbar = plt.colorbar(sm)
    cbar.set_label('Color', rotation=270, labelpad=25)

    plt.yticks(np.arange(0, 1, 0.1))
    plt.ylabel('ppscore')
    plt.xlabel('Features')
