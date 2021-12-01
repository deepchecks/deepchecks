"""The class performance imbalance check."""
#pylint: disable=inconsistent-quotes

import typing as t
from functools import partial
from collections import defaultdict

import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, get_scorer
from sklearn.utils.multiclass import unique_labels as get_unique_labels

from deepchecks import SingleDatasetBaseCheck, Dataset, CheckResult, ConditionResult
from deepchecks.utils.metrics import task_type_validation, ModelType
from deepchecks.utils.strings import format_percent
from deepchecks.errors import DeepchecksValueError


__all__ = ['ClassPerformanceImbalanceCheck']


ScorerFunc = t.Callable[
    [object, pd.DataFrame, pd.Series], # model, features, labels
    t.Dict[t.Hashable, t.Union[float, int]] # score for each label
]

AlternativeScorer = t.Union[str, ScorerFunc]


CP = t.TypeVar('CP', bound='ClassPerformanceImbalanceCheck')


class ClassPerformanceImbalanceCheck(SingleDatasetBaseCheck):
    """Visualize class imbalance by displaying the difference between class score values.

    Args:
        alternative_scorers (Mapping[str, Union[str, Callable]]):
            An optional dictionary of scorer name or scorer functions

    Raises:
        DeepchecksValueError:
            if provided dict of scorers is emtpy;
            if one of the entries of the provided scorers dict contains not callable value;
    """

    def __init__(
        self,
        alternative_scorers: t.Optional[t.Mapping[str, AlternativeScorer]] = None
    ):
        super().__init__()
        self.alternative_scorers: t.Optional[t.Mapping[str, ScorerFunc]] = None

        if alternative_scorers is not None and len(alternative_scorers) == 0:
            raise DeepchecksValueError('alternative_scorers - expected to receive not empty dict of scorers!')

        elif alternative_scorers is not None:
            self.alternative_scorers = {}

            for name, scorer in alternative_scorers.items():
                if isinstance(scorer, t.Callable):
                    self.alternative_scorers[name] = scorer
                elif isinstance(scorer, str):
                    self.alternative_scorers[name] = get_scorer(scorer)
                else:
                    raise DeepchecksValueError(
                        f"alternative_scorers - expected to receive 'Mapping[str, Callable]' but got "
                        f"'Mapping[str, {type(scorer).__name__}]'!"
                    )

    def run(
        self,
        dataset: Dataset,
        model: t.Any # TODO: find more precise type for model
    ) -> CheckResult:
        """Run Check.

        Args:
            dataset (Dataset): a dataset object
            model (BaseEstimator): A scikit-learn-compatible fitted estimator instance

        Returns:
            CheckResult: value is a dictionary of labels, value of each key is another dictionary
                with calculated scores

        Raises:
            DeepchecksValueError:
                if task type is not binary or multi-class;
                if dataset does not have a label column or the label dtype is unsupported;
                if provided dataset is empty;
                if provided dataset does not have features columns;
        """
        return self._class_performance_imbalance(dataset, model)

    def _class_performance_imbalance(
        self,
        dataset: Dataset,
        model: t.Any # TODO: find more precise type for model
    ) -> CheckResult:
        check_name = type(self).__name__
        expected_model_types = [ModelType.BINARY, ModelType.MULTICLASS]

        Dataset.validate_dataset(dataset, check_name)
        dataset.validate_label(check_name)
        dataset.validate_features(check_name)
        task_type_validation(model, dataset, expected_model_types, check_name)

        labels = t.cast(pd.Series, dataset.label_col())
        features = t.cast(pd.DataFrame, dataset.features_columns())

        if self.alternative_scorers is not None:
            df = pd.DataFrame.from_dict(self._execute_alternative_scorers(
                model, features, labels
            ))
        else:
            y_true = labels
            y_pred = model.predict(features)
            unique_labels = get_unique_labels(labels, y_pred)
            df = pd.DataFrame.from_dict({
                name: dict(zip(
                    unique_labels,
                    t.cast(t.Iterable, scorer_func(y_true, y_pred, labels=unique_labels))
                ))
                for name, scorer_func in self._get_default_scorers().items()
            })

        def display():
            title = (
                'Class Performance Imbalance Check for binary data'
                if len(labels) == 2
                else 'Class Performance Imbalance Check for multi-class data'
            )

            df.transpose().plot.bar(
                title=title,
                backend='matplotlib',
                xlabel='Score',
                ylabel='Values'
            )

        return CheckResult(
            value=df.transpose().to_dict(),
            header='Class Performance Imbalance',
            check=type(self),
            display=display
        )

    def _get_default_scorers(self) -> t.Dict[str, t.Callable[..., np.ndarray]]:
        return {
            'Accuracy': partial(recall_score, zero_division=0, average=None),
            'Precision': partial(precision_score, zero_division=0, average=None),
            'Recall': partial(recall_score, zero_division=0, average=None)
        }

    def _execute_alternative_scorers(
        self,
        model: t.Any,
        features: pd.DataFrame,
        labels: pd.Series,
    ) -> t.Dict[str, t.Dict[t.Hashable, t.Union[int, float]]]:
        check_name = type(self).__name__
        scorers = t.cast(t.Mapping[str, ScorerFunc], self.alternative_scorers)
        result: t.Dict[str, t.Dict[t.Hashable, t.Union[int, float]]] = {}

        for scorer_name, scorer_func in scorers.items():
            scorer_result = scorer_func(model, features, labels)

            if not isinstance(scorer_result, dict):
                result_type = type(scorer_result).__name__
                raise DeepchecksValueError(
                    f"Check {check_name} expecting that alternative scorer '{scorer_name}' will return "
                    f"not empty instance of 'Mapping[Hashable, float|int]', but got {result_type}"
                )

            if len(scorer_result) == 0:
                raise DeepchecksValueError(
                    f"Check {check_name} expecting that alternative scorer '{scorer_name}' will return "
                    "not empty instance of 'Mapping[Hashable, float|int]'"
                )

            incorrect_values = [v for v in scorer_result.values() if not isinstance(v, (int, float))]

            if len(incorrect_values) != 0:
                value_type = type(incorrect_values[0]).__name__
                raise DeepchecksValueError(
                    f"Check {check_name} expecting that alternative scorer '{scorer_name}' will return "
                    "not empty instance of 'Mapping[Hashable, float|int]', "
                    f"but got 'Mapping[Hashable, {value_type}]'"
                )

            result[scorer_name] = scorer_result

        return result

    def add_condition_ratio_difference_not_greater_than(self: CP, threshold: float = 0.3) -> CP:
        """Add condition.

        Verifying that relative ratio difference
        between highest-class and lowest-class is not greater than 'threshold'.

        Args:
            threshold: ratio difference threshold

        Returns:
            Self: instance of 'ClassPerformanceImbalanceCheck' or it subtype
        """

        def condition(check_result: t.Dict[str, t.Dict[t.Hashable, float]]) -> ConditionResult:
            data = t.cast(
                t.Dict[str, t.Dict[t.Hashable, float]],
                pd.DataFrame.from_dict(check_result).transpose().to_dict()
            )

            result = defaultdict(dict)

            # For each score calculate: (highest-class - lowest-class)/highest-class
            for score_name, classes_values in data.items():
                getval = lambda it: it[1]
                lowest_class_name, min_value = min(classes_values.items(), key=getval)
                highest_class_name, max_value = max(classes_values.items(), key=getval)

                relative_difference = abs((min_value - max_value) / max_value)

                if relative_difference >= threshold:
                    result[score_name][(lowest_class_name, highest_class_name)] = relative_difference

            if len(result) == 0:
                return ConditionResult(True)

            details = '\n'.join([
                f'Score: {score_name}, lowest class: {lowest_class_name}, highest class: {highest_class_name};'
                for score_name, difference in result.items()
                for ((lowest_class_name, highest_class_name), _) in difference.items()
            ])

            details = (
                'Relative ratio difference between highest and lowest classes is greater '
                f'than {format_percent(threshold)}:\n{details}'
            )

            return ConditionResult(False, details=details)

        return self.add_condition(
            name=f'Relative ratio difference is not greater than {format_percent(threshold)}',
            condition_func=condition
        )
