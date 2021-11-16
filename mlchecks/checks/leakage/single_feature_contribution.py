"""The single_feature_contribution check module."""
import typing as t

import mlchecks.ppscore as pps
from mlchecks.plot_utils import create_colorbar_barchart_for_check
from mlchecks.utils import MLChecksValueError
from mlchecks import CheckResult, Dataset, SingleDatasetBaseCheck, ConditionCategory, ConditionResult


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

    def _condition_factory(
        self,
        var: float,
        features: t.Optional[t.Sequence[str]],
        category: ConditionCategory,
        success_message: str,
        failure_message: str,
        operator: t.Callable[[float, float], bool]
    ) -> t.Callable[[t.Dict[str, float]], ConditionResult]:

        if features is not None and len(features) == 0:
            raise MLChecksValueError("sequence of 'features' cannot be empty!")
        
        def condition(value: t.Dict[str, float]) -> ConditionResult:
            nonlocal features
            
            if features is None:
                features_to_check = set(value.keys())
            else:
                features_to_check = set(features)
                available_features = set(value.keys())
                features_difference = features_to_check.difference(available_features)

                if len(features_difference) != 0:
                    raise MLChecksValueError(f"unknown features - {features_difference}")

            all_features = []
            failed_features = []

            for feature_name, pps in value.items():
                feature_repr = f"{feature_name} (pps: {pps})"
                all_features.append(feature_repr)
                if feature_name in features_to_check and operator(pps, var) is True:
                    failed_features.append(feature_repr)
            
            details_template_vars = {
                "var": var,
                "all_features": all_features,
                "failed_features": failed_features
            }

            passed = len(failed_features) == 0
            
            return ConditionResult(
                is_pass=passed,
                category=category,
                details=(
                    success_message.format(**details_template_vars)
                    if passed
                    else failure_message.format(**details_template_vars)
                )
            )
        
        return condition
    
    def add_condition_feature_pps_lt_than(
        self, 
        var: float,
        *,
        features: t.Optional[t.Sequence[str]] = None,
        category = ConditionCategory.FAIL,
        success_message = "Condition passed. All features pps >= {var}.",
        failure_message = "Condition failed. Next features pps < {var}: {failed_features}",
        name = "Features PPS lower bound (operator: lt)"
    ):
        """
        Add condition that will check that pps of the specified feature(s) is not < X.

        If `features` parameter is `None`, condition will be applied to all features.

        Args:
            var
            features: list of features to check
            category: condition category
            success_message: condition details template in case of success
            failure_message: condition details template in case of failure
            name: condition name
        
        Raises:
            MLChecksValueError: if empty list of features was passed to the method
        
        Condition Raises:
            MLChecksValueError: if `features` list contains unknown feature
        """
        self.add_condition(
            name=name, 
            condition_func=self._condition_factory(
                var,
                features,
                category, 
                success_message, 
                failure_message,
                operator=lambda pps, var: pps < var
            )
        )
    
    def add_condition_feature_pps_le_than(
        self, 
        var: float,
        *,
        features: t.Optional[t.Sequence[str]] = None,
        category = ConditionCategory.FAIL,
        success_message = "Condition passed. All features pps > {var}.",
        failure_message = "Condition failed. Next features pps <= {var}: {failed_features}",
        name = "Features PPS lower bound (operator: le)"
    ):
        """
        Add condition that will check that pps of the specified feature(s) is not <= X.

        If `features` parameter is `None`, condition will be applied to all features.

        Args:
            var
            features: list of features to check
            category: condition category
            success_message: condition details in case of success
            failure_message: condition details in case of failure
            name: condition name
        
        Raises:
            MLChecksValueError: if empty list of features was passed to the method
        
        Condition Raises:
            MLChecksValueError: if `features` list contains unknown feature
        """
        self.add_condition(
            name=name, 
            condition_func=self._condition_factory(
                var,
                features,
                category,
                success_message,
                failure_message,
                operator=lambda pps, var: pps <= var
            )
        )

    def add_condition_feature_pps_gt_than(
        self, 
        var: float,
        *,
        features: t.Optional[t.Sequence[str]] = None,
        category = ConditionCategory.FAIL,
        success_message = "Condition passed. All features pps <= {var}.",
        failure_message = "Condition failed. Next features pps > {var}: {failed_features}",
        name = "Features PPS upper bound (operator: gt)"
    ):
        """
        Add condition that will check that pps of the specified feature(s) is not > X.

        If `features` parameter is `None`, condition will be applied to all features.

        Args:
            var
            features: list of features to check
            category: condition category
            success_message: condition details in case of success
            failure_message: condition details in case of failure
            name: condition name
        
        Raises:
            MLChecksValueError: if empty list of features was passed to the method
        
        Condition Raises:
            MLChecksValueError: if `features` list contains unknown feature        
        """
        self.add_condition(
            name=name, 
            condition_func=self._condition_factory(
                var,
                features,
                category,
                success_message,
                failure_message,
                operator=lambda pps, var: pps > var
            )
        )

    def add_condition_feature_pps_ge_than(
        self, 
        var: float,
        *,
        features: t.Optional[t.Sequence[str]] = None,
        category = ConditionCategory.FAIL,
        success_message = "Condition passed. All features pps < {var}.",
        failure_message = "Condition failed. Next features pps >= {var}: {failed_features}",
        name = "Features PPS upper bound (operator: ge)"
    ):
        """
        Add condition that will check that pps of the specified feature(s) is not > X.

        If `features` parameter is `None`, condition will be applied to all features.

        Args:
            var
            features: list of features to check
            category: condition category
            success_message: condition details in case of success
            failure_message: condition details in case of failure
            name: condition name
        
        Raises:
            MLChecksValueError: if empty list of features was passed to the method
        
        Condition Raises:
            MLChecksValueError: if `features` list contains unknown feature        
        """
        self.add_condition(
            name=name, 
            condition_func=self._condition_factory(
                var,
                features,
                category,
                success_message,
                failure_message,
                operator=lambda pps, var: pps >= var
            )
        )