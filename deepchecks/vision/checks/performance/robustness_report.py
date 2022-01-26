"""Module containing performance report check."""
from typing import Callable, TypeVar, Dict, cast, List, Optional, Union

import albumentations.core.composition
import albumentations as A
import pandas as pd
import plotly.express as px
from ignite.metrics import Metric

from deepchecks import CheckResult, TrainTestBaseCheck, ConditionResult
from deepchecks.errors import DeepchecksValueError
from deepchecks.utils.metrics import MULTICLASS_SCORERS_NON_AVERAGE
from deepchecks.utils.strings import format_percent, format_number
from deepchecks.vision import VisionDataset
from deepchecks.vision.base.vision_dataset import TaskType
from deepchecks.vision.utils.metrics import get_scorers_list, task_type_check, calculate_metrics

__all__ = ['RobustnessReport']

from deepchecks.vision.utils.validation import model_type_validation

PR = TypeVar('PR', bound='RobustnessReport')


class RobustnessReport(TrainTestBaseCheck):
    """
    Check several image enhancements for model robustness.
    Args:
        alternative_metrics (List[Metric], default None):
            A list of ignite.Metric objects whose score should be used. If None are given, use the default metrics.
    """

    def __init__(self, alternative_metrics: Optional[List[Metric]] = None,
                 prediction_extract: Optional[Callable] = None,
                 augmentations: Optional[Union[A.Compose, List[A.BasicTransform]]] = None):
        super().__init__()
        self.alternative_metrics = alternative_metrics
        # Now we duplicate the val_dataloader and create an augmented one
        # Note that p=1.0 since we want to apply those to entire dataset
        # To use albumentations I need to do this:
        self.augmentaions = [
            A.RandomBrightnessContrast(p=1.0),
            A.ShiftScaleRotate(p=1.0),
            A.HueSaturationValue(p=1.0),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=1.0),
        ]

    def run(self, baseline_dataset: VisionDataset, augmented_dataset: VisionDataset, model=None) -> CheckResult:
        """Run check.

        Args:
            baseline_dataset (VisionDataset): a VisionDataset object
            augmented_dataset (VisionDataset): a VisionDataset object
            model : A torch model supporting inference in eval mode.

        Returns:
            CheckResult: value is dictionary in format 'score-name': score-value
        """
        return self._performance_report(baseline_dataset, augmented_dataset, model)

    def _performance_report(self, baseline_dataset: VisionDataset, augmented_dataset: VisionDataset, model):
        VisionDataset.validate_dataset(baseline_dataset)
        VisionDataset.validate_dataset(augmented_dataset)
        baseline_dataset.validate_label()
        augmented_dataset.validate_label()
        baseline_dataset.validate_shared_label(augmented_dataset)
        model_type_validation(model)

        task_type = task_type_check(model, baseline_dataset)

        # Get default scorers if no alternative, or validate alternatives
        scorers = get_scorers_list(model, augmented_dataset, baseline_dataset.get_num_classes(),
                                   self.alternative_metrics)
        datasets = {'Train': baseline_dataset, 'Test': augmented_dataset}

        plot_x_axis = 'Class'
        if task_type in (TaskType.CLASSIFICATION, TaskType.OBJECT_DETECTION):
            results_df = self._performance_report_inner_loop(baseline_dataset, datasets, model, scorers)

        else:
            return NotImplementedError('only works for classification ATM')

        fig = px.histogram(
            results_df,
            x=plot_x_axis,
            y='Value',
            color='Dataset',
            barmode='group',
            facet_col='Metric',
            facet_col_spacing=0.05,
            hover_data=['Number of samples']
        )

        if task_type == TaskType.CLASSIFICATION:
            fig.update_xaxes(tickprefix='Class ', tickangle=60)

        fig = (
            fig.update_xaxes(title=None, type='category')
                .update_yaxes(title=None, matches=None)
                .for_each_annotation(lambda a: a.update(text=a.text.split('=')[-1]))
                .for_each_yaxis(lambda yaxis: yaxis.update(showticklabels=True))
        )

        return CheckResult(
            results_df,
            header='Robustness Report',
            display=fig
        )

    def _performance_report_inner_loop(self, baseline_dataset, datasets, model, scorers):
        classes = baseline_dataset.get_samples_per_class().keys()
        results = []
        for dataset_name, dataset in datasets.items():
            n_samples = dataset.get_samples_per_class()
            results.extend(
                [dataset_name, class_name, name, class_score, n_samples[class_name]]
                for name, score in calculate_metrics(list(scorers.values()), dataset, model,
                                                     prediction_extract=self.prediction_extract).items()
                # scorer returns numpy array of results with item per class
                for class_score, class_name in zip(score, classes)
            )
        results_df = pd.DataFrame(results, columns=['Dataset', 'Class', 'Metric', 'Value', 'Number of samples']
                                  ).sort_values(by=['Class'])
        return results_df

    def add_condition_test_performance_not_less_than(self: PR, min_score: float) -> PR:
        """Add condition - metric scores are not less than given score.

        Args:
            min_score (float): Minimal score to pass.
        """

        def condition(check_result: pd.DataFrame):
            not_passed = check_result.loc[check_result['Value'] < min_score]
            not_passed_test = check_result.loc[check_result['Dataset'] == 'Test']
            if len(not_passed):
                details = f'Found metrics with scores below threshold:\n' \
                          f'{not_passed_test[["Class", "Metric", "Value"]].to_dict("records")}'
                return ConditionResult(False, details)
            return ConditionResult(True)

        return self.add_condition(f'Scores are not less than {min_score}', condition)

    def add_condition_train_test_relative_degradation_not_greater_than(self: PR, threshold: float = 0.1) -> PR:
        """Add condition that will check that test performance is not degraded by more than given percentage in train.

        Args:
            threshold: maximum degradation ratio allowed (value between 0 and 1)
        """

        def _ratio_of_change_calc(score_1, score_2):
            if score_1 == 0:
                if score_2 == 0:
                    return 0
                return threshold + 1
            return (score_1 - score_2) / abs(score_1)

        def condition(check_result: pd.DataFrame) -> ConditionResult:
            test_scores = check_result.loc[check_result['Dataset'] == 'Test']
            train_scores = check_result.loc[check_result['Dataset'] == 'Train']

            if check_result.get('Class') is not None:
                classes = check_result['Class'].unique()
            else:
                classes = None
            explained_failures = []
            if classes is not None:
                for class_name in classes:
                    test_scores_class = test_scores.loc[test_scores['Class'] == class_name]
                    train_scores_class = train_scores.loc[train_scores['Class'] == class_name]
                    test_scores_dict = dict(zip(test_scores_class['Metric'], test_scores_class['Value']))
                    train_scores_dict = dict(zip(train_scores_class['Metric'], train_scores_class['Value']))
                    # Calculate percentage of change from train to test
                    diff = {score_name: _ratio_of_change_calc(score, test_scores_dict[score_name])
                            for score_name, score in train_scores_dict.items()}
                    failed_scores = [k for k, v in diff.items() if v > threshold]
                    if failed_scores:
                        for score_name in failed_scores:
                            explained_failures.append(f'{score_name} for class {class_name} '
                                                      f'(train={format_number(train_scores_dict[score_name])} '
                                                      f'test={format_number(test_scores_dict[score_name])})')
            else:
                test_scores_dict = dict(zip(test_scores['Metric'], test_scores['Value']))
                train_scores_dict = dict(zip(train_scores['Metric'], train_scores['Value']))
                # Calculate percentage of change from train to test
                diff = {score_name: _ratio_of_change_calc(score, test_scores_dict[score_name])
                        for score_name, score in train_scores_dict.items()}
                failed_scores = [k for k, v in diff.items() if v > threshold]
                if failed_scores:
                    for score_name in failed_scores:
                        explained_failures.append(f'{score_name}: '
                                                  f'train={format_number(train_scores_dict[score_name])}, '
                                                  f'test={format_number(test_scores_dict[score_name])}')
            if explained_failures:
                message = '\n'.join(explained_failures)
                return ConditionResult(False, message)
            else:
                return ConditionResult(True)

        return self.add_condition(f'Train-Test scores relative degradation is not greater than {threshold}',
                                  condition)

    def add_condition_class_performance_imbalance_ratio_not_greater_than(
            self: PR,
            threshold: float = 0.3,
            score: str = None
    ) -> PR:
        """Add condition.

        Verifying that relative ratio difference
        between highest-class and lowest-class is not greater than 'threshold'.

        Args:
            threshold: ratio difference threshold
            score: limit score for condition

        Returns:
            Self: instance of 'ClassPerformance' or it subtype

        Raises:
            DeepchecksValueError:
                if unknown score function name were passed;
        """
        if score is None:
            score = next(iter(MULTICLASS_SCORERS_NON_AVERAGE))

        def condition(check_result: pd.DataFrame) -> ConditionResult:
            if score not in set(check_result['Metric']):
                raise DeepchecksValueError(f'Data was not calculated using the scoring function: {score}')

            datasets_details = []
            for dataset in ['Test', 'Train']:
                data = check_result.loc[check_result['Dataset'] == dataset].loc[check_result['Metric'] == score]

                min_value_index = data['Value'].idxmin()
                min_row = data.loc[min_value_index]
                min_class_name = min_row['Class']
                min_value = min_row['Value']

                max_value_index = data['Value'].idxmax()
                max_row = data.loc[max_value_index]
                max_class_name = max_row['Class']
                max_value = max_row['Value']

                relative_difference = abs((min_value - max_value) / max_value)

                if relative_difference >= threshold:
                    details = (
                        f'Relative ratio difference between highest and lowest in {dataset} dataset '
                        f'classes is {format_percent(relative_difference)}, using {score} metric. '
                        f'Lowest class - {min_class_name}: {format_number(min_value)}; '
                        f'Highest class - {max_class_name}: {format_number(max_value)}'
                    )
                    datasets_details.append(details)
            if datasets_details:
                return ConditionResult(False, details='\n'.join(datasets_details))
            else:
                return ConditionResult(True)

        return self.add_condition(
            name=(
                f'Relative ratio difference between labels \'{score}\' score '
                f'is not greater than {format_percent(threshold)}'
            ),
            condition_func=condition
        )
