"""The single_feature_contribution check module."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import ppscore as pps

from mlchecks import CheckResult, Dataset, SingleDatasetBaseCheck, TrainValidationBaseCheck

__all__ = ['single_feature_contribution', 'single_feature_contribution_train_validation',
           'SingleFeatureContribution', 'SingleFeatureContributionTrainValidation']


def single_feature_contribution(dataset: Dataset, ppscore_params=None):
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
    self = single_feature_contribution
    Dataset.validate_dataset(dataset, self.__name__)
    dataset.validate_label(self.__name__)
    ppscore_params = ppscore_params or {}

    relevant_columns = dataset.features() + [dataset.label_name()]
    df_pps = pps.predictors(df=dataset.data[relevant_columns], y=dataset.label_name(), random_seed=42, **ppscore_params)
    df_pps = df_pps.set_index('x', drop=True)
    s_ppscore = df_pps['ppscore']

    def plot():
        # Create graph:
        create_colorbar_barchart_for_check(x=s_ppscore.index, y=s_ppscore.values)

    text = ['The PPS represents the ability of a feature to single-handedly predict another feature or label.',
            'A high PPS (close to 1) can mean that this feature\'s success in predicting the label is actually due to '
            'data',
            'leakage - meaning that the feature holds information that is based on the label to begin with.']

    return CheckResult(value=s_ppscore.to_dict(), display=[plot, *text], check=self)


def single_feature_contribution_train_validation(train_dataset: Dataset, validation_dataset: Dataset,
                                                 ppscore_params=None, n_show_top: int = 5):
    """
    Return the difference in PPS (Predictive Power Score) of all features between train and validation datasets.

    The PPS represents the ability of a feature to single-handedly predict another feature or label.
    A high PPS (close to 1) can mean that this feature's success in predicting the label is actually due to data
    leakage - meaning that the feature holds information that is based on the label to begin with.

    When we compare train PPS to validation PPS, A high difference can strongly indicate leakage, as a feature that was
    "powerful" in train but not in validation can be explained by leakage in train that does not affect a new dataset.

    Uses the ppscore package - for more info, see https://github.com/8080labs/ppscore

    Args:
        train_dataset (Dataset): The training dataset object. Must contain a label
        validation_dataset (Dataset): The validation dataset object. Must contain a label
        ppscore_params (dict): dictionary of additional parameters for the ppscore predictor function
        n_show_top (int): Number of features to show, sorted by the magnitude of difference in PPS

    Returns:
        CheckResult:
            value is a dictionary with PPS difference per feature column.
            data is a bar graph of the PPS of each feature.

    Raises:
        MLChecksValueError: If the object is not a Dataset instance with a label

    """
    self = single_feature_contribution_train_validation
    train_dataset = Dataset.validate_dataset(train_dataset, self.__name__)
    train_dataset.validate_label(self.__name__)
    validation_dataset = Dataset.validate_dataset(validation_dataset, self.__name__)
    validation_dataset.validate_label(self.__name__)
    features_names = train_dataset.validate_shared_features(validation_dataset, self.__name__)
    label_name = train_dataset.validate_shared_label(validation_dataset, self.__name__)
    ppscore_params = ppscore_params or {}

    relevant_columns = features_names + [label_name]
    df_pps_train = pps.predictors(df=train_dataset.data[relevant_columns], y=train_dataset.label_name(), random_seed=42,
                                  **ppscore_params)
    df_pps_validation = pps.predictors(df=validation_dataset.data[relevant_columns], y=validation_dataset.label_name(),
                                       random_seed=42, **ppscore_params)
    s_pps_train = df_pps_train.set_index('x', drop=True)['ppscore']
    s_pps_validation = df_pps_validation.set_index('x', drop=True)['ppscore']

    s_difference = s_pps_train - s_pps_validation
    s_difference = s_difference.apply(lambda x: 0 if x < 0 else x)
    s_difference = s_difference.sort_values(ascending=False).head(n_show_top)

    def plot():
        # Create graph:
        create_colorbar_barchart_for_check(x=s_difference.index, y=s_difference.values, ylabel='PPS Difference')

    text = ['The PPS represents the ability of a feature to single-handedly predict another feature or label.',
            'A high PPS (close to 1) can mean that this feature\'s success in predicting the label is actually due to '
            'data',
            'leakage - meaning that the feature holds information that is based on the label to begin with.',
            '',
            'When we compare train PPS to validation PPS, A high difference can strongly indicate leakage, as a '
            'feature',
            'that was powerful in train but not in validation can be explained by leakage in train that is not '
            'relevant to a new dataset.']

    return CheckResult(value=s_difference.to_dict(), display=[plot, *text], check=self,
                       header='Single Feature Contribution Train-Validation')


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
            the output of the single_feature_contribution check
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
        train_dataset (Dataset): The training dataset object. Must contain a label
        validation_dataset (Dataset): The validation dataset object. Must contain a label
            model: any = None - not used in the check

        Returns:
            the output of the single_feature_contribution_train_validation check
        """
        return single_feature_contribution_train_validation(train_dataset=train_dataset,
                                                            validation_dataset=validation_dataset,
                                                            ppscore_params=self.params.get('ppscore_params'))


# Utils:


def create_colorbar_barchart_for_check(x: np.array, y: np.array, ylabel: str='PPS'):
    fig, ax = plt.subplots(figsize=(15, 4))  # pylint: disable=unused-variable

    my_cmap = plt.cm.get_cmap('RdYlGn_r')
    colors = my_cmap(list(y))
    rects = ax.bar(x, y, color=colors)  # pylint: disable=unused-variable

    sm = ScalarMappable(cmap=my_cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])

    cbar = plt.colorbar(sm)
    cbar.set_label('Color', rotation=270, labelpad=25)

    plt.yticks(np.arange(0, 1, 0.1))
    plt.ylabel(ylabel)
    plt.xlabel('Features')
