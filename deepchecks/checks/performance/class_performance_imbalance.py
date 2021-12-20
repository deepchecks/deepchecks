# ----------------------------------------------------------------------------
# Copyright (C) 2021 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""The class performance imbalance check."""
#pylint: disable=inconsistent-quotes
import typing as t

import pandas as pd
import numpy as np
from sklearn.metrics import make_scorer, f1_score
from sklearn.utils.multiclass import unique_labels as get_unique_labels

from deepchecks import SingleDatasetBaseCheck, Dataset, CheckResult, ConditionResult
from deepchecks.utils.metrics import task_type_validation, ModelType, initialize_user_scorers, get_scorers
from deepchecks.utils.strings import format_percent
from deepchecks.errors import DeepchecksValueError


__all__ = ['ClassPerformanceImbalance']


ScorerFunc = t.Callable[
    [object, pd.DataFrame, pd.Series], # model, features, labels
    np.ndarray # scores
]

AlternativeScorer = t.Union[str, ScorerFunc]


CP = t.TypeVar('CP', bound='ClassPerformanceImbalance')


class ClassPerformanceImbalance(SingleDatasetBaseCheck):
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
        if alternative_scorers is not None:
            self.alternative_scorers = initialize_user_scorers(alternative_scorers)

    def run(
        self,
        dataset: Dataset,
        model: t.Any
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
        expected_model_types = [ModelType.BINARY, ModelType.MULTICLASS]

        Dataset.validate_dataset(dataset)
        dataset.validate_label()
        dataset.validate_features()
        task_type_validation(model, dataset, expected_model_types)

        labels = t.cast(pd.Series, dataset.label_col)
        features = t.cast(pd.DataFrame, dataset.features_columns)

        unique_labels = get_unique_labels(labels)
        scorers = get_scorers(model, dataset, self.alternative_scorers, average=False)
        # In case of default scorers adds F1
        if self.alternative_scorers is None:
            scorers['F1'] = make_scorer(f1_score, average=None)

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
            display=display
        )

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

    def add_condition_ratio_difference_not_greater_than(
        self: CP,
        threshold: float = 0.3,
        score: str = 'F1'
    ) -> CP:
        """Add condition.

        Verifying that relative ratio difference
        between highest-class and lowest-class is not greater than 'threshold'.

        Args:
            threshold: ratio difference threshold

        Returns:
            Self: instance of 'ClassPerformanceImbalance' or it subtype

        Raises:
            DeepchecksValueError:
                if unknown score function name were passed;
        """
        if self.alternative_scorers:
            scorers = set(self.alternative_scorers.keys())
        else:
            scorers = ['Precision', 'Recall', 'F1']

        if score not in scorers:
            raise DeepchecksValueError(f'Unknown score function  - {score}')

        def condition(check_result: t.Dict[str, t.Dict[t.Hashable, float]]) -> ConditionResult:
            data = t.cast(
                t.Dict[str, t.Dict[t.Hashable, float]],
                pd.DataFrame.from_dict(check_result).transpose().to_dict()
            )
            data = [
                classes_values
                for score_name, classes_values in data.items()
                if score_name == score
            ]

            if len(data) == 0:
                raise DeepchecksValueError(f'Expected that check result will contain next score - {score}')

            classes_values = next(iter(data))
            getval = lambda it: it[1]
            lowest_class_name, min_value = min(classes_values.items(), key=getval)
            highest_class_name, max_value = max(classes_values.items(), key=getval)
            relative_difference = abs((min_value - max_value) / max_value)

            if relative_difference >= threshold:
                details = (
                    'Relative ratio difference between highest and lowest '
                    f'classes is greater than {format_percent(threshold)}. '
                    f'Score: {score}, lowest class: {lowest_class_name}, highest class: {highest_class_name};'
                )
                return ConditionResult(False, details=details)
            else:
                return ConditionResult(True)

        return self.add_condition(
            name=(
                f"Relative ratio difference between labels '{score}' score "
                f"is not greater than {format_percent(threshold)}"
            ),
            condition_func=condition
        )
