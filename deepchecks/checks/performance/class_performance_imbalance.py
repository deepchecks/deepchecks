"""The class performance imbalance check."""
#pylint: disable=inconsistent-quotes

import typing as t
from collections import defaultdict

import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, get_scorer, make_scorer
from sklearn.utils.multiclass import unique_labels as get_unique_labels

from deepchecks import SingleDatasetBaseCheck, Dataset, CheckResult, ConditionResult
from deepchecks.utils.metrics import task_type_validation, ModelType
from deepchecks.utils.strings import format_percent
from deepchecks.errors import DeepchecksValueError


__all__ = ['ClassPerformanceImbalanceCheck']


ScorerFunc = t.Callable[
    [object, pd.DataFrame, pd.Series], # model, features, labels
    np.ndarray # scores
]

AlternativeScorer = t.Union[str, ScorerFunc]


CP = t.TypeVar('CP', bound='ClassPerformanceImbalanceCheck')


class ClassPerformanceImbalanceCheck(SingleDatasetBaseCheck):
    """Visualize class imbalance by displaying the difference between class score values.

    Args:
        alternative_scorers (Mapping[str, Union[str, Callable]]):
            An optional dictionary of scorer name or scorer functions.
            Important, user-defined scorer alternative functions must return
            array in sorted order, to match `sorted(unique_labels)`.

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

        unique_labels = get_unique_labels(labels)
        scorers = self.alternative_scorers or self._get_default_scorers()

        scorer_results = (
            (scorer_name, scorer_func(model, features, labels))
            for scorer_name, scorer_func in scorers.items()
        )

        df = pd.DataFrame.from_dict({
            scorer_name: dict(zip(unique_labels, score))
            for scorer_name, score in self._validate_results(scorer_results, len(unique_labels))
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
        # TODO: use `get_metrics_list` from utils package
        # but first we need to refactor it to accept 'average' argument
        return {
            'Accuracy': make_scorer(recall_score, zero_division=0, average=None),
            'Precision': make_scorer(precision_score, zero_division=0, average=None),
            'Recall': make_scorer(recall_score, zero_division=0, average=None)
        }

    def _validate_results(
        self,
        results: t.Iterable[t.Tuple[str, object]],
        number_of_classes: int
    ) -> t.Iterator[t.Tuple[str, np.ndarray]]:
        # We need to be sure that user-provided scorers returned value with correct
        # datatype, otherwise we need to raise an exception with an informative message
        expected_types = t.cast(
            str,
            np.typecodes['AllInteger'] + np.typecodes['AllFloat'] # type: ignore
        )
        message = (
            f"Check '{type(self).__name__}' expecting that scorer "
            "'{scorer_name}' will return an instance of numpy array with "
            f"items of type int|float and with shape ({number_of_classes},)! {{additional}}."
        )

        for scorer_name, score in results:
            if not isinstance(score, np.ndarray):
                raise DeepchecksValueError(message.format(
                    scorer_name=scorer_name,
                    additional=f"But got instance of '{type(score).__name__}'"
                ))
            if score.dtype.kind not in expected_types:
                raise DeepchecksValueError(message.format(
                    scorer_name=scorer_name,
                    additional=f"But got array of '{str(score.dtype)}'"
                ))
            if len(score) != number_of_classes:
                raise DeepchecksValueError(message.format(
                    scorer_name=scorer_name,
                    additional=f"But got array with shape ({len(score)},)"
                ))
            yield scorer_name, score

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
