"""Module containing robustness report check."""
from typing import Callable, TypeVar, List, Optional, Union
import albumentations as A
import pandas as pd
import plotly.express as px
from ignite.metrics import Metric

from deepchecks import CheckResult, TrainTestBaseCheck
from deepchecks.errors import DeepchecksValueError
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
        self.prediction_extract = prediction_extract
        # Now we duplicate the val_dataloader and create an augmented one
        # Note that p=1.0 since we want to apply those to entire dataset
        # To use albumentations I need to do this:
        if augmentations is None:
            augmentations = [
                A.RandomBrightnessContrast(p=1.0),
                A.ShiftScaleRotate(p=1.0),
                A.HueSaturationValue(p=1.0),
                A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=1.0),
            ]
        elif isinstance(augmentations, A.Compose):
            augmentations = augmentations.transforms.transforms
        self.augmentations = augmentations

    def run(self, *, baseline_dataset: VisionDataset, augmented_dataset: VisionDataset, model=None) -> CheckResult:
        """Run check.

        Args:
            baseline_dataset (VisionDataset): a VisionDataset object
            augmented_dataset (VisionDataset): a VisionDataset object
            model : A torch model supporting inference in eval mode.

        Returns:
            CheckResult: value is dictionary in format 'score-name': score-value
        """
        return self._robustness_report(baseline_dataset, augmented_dataset, model)

    def _robustness_report(self, baseline_dataset: VisionDataset, augmented_dataset: VisionDataset, model):
        VisionDataset.validate_dataset(baseline_dataset)
        VisionDataset.validate_dataset(augmented_dataset)
        if set(baseline_dataset.get_samples_per_class().keys()) != set(augmented_dataset.get_samples_per_class().keys()):
            raise DeepchecksValueError("Datasets must share class count")
        if not all([set(baseline_dataset.get_samples_per_class()[k]) == set(augmented_dataset.get_samples_per_class()[k])
                    for k in baseline_dataset.get_samples_per_class().keys()]):
            raise DeepchecksValueError("Dataset must have same numnber of examples per class")
        baseline_dataset.validate_label()
        augmented_dataset.validate_label()
        baseline_dataset.validate_shared_label(augmented_dataset)
        model_type_validation(model)

        task_type = task_type_check(model, baseline_dataset)

        # Get default scorers if no alternative, or validate alternatives
        scorers = get_scorers_list(model, augmented_dataset, baseline_dataset.get_num_classes(),
                                   self.alternative_metrics)
        if task_type in (TaskType.CLASSIFICATION, TaskType.OBJECT_DETECTION):
            results_df = self._report_inner_loop(baseline_dataset, augmented_dataset, model, scorers)
        else:
            return NotImplementedError('only works for classification ATM')

        plot_x_axis = 'Class'
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

    def _report_inner_loop(self, baseline_dataset, augmented_dataset, model, scorers):
        # We have already checked and both
        classes = baseline_dataset.get_samples_per_class().keys()
        n_samples = baseline_dataset.get_samples_per_class()
        def evaluate_dataset(dataset):
            return (
                [class_name, name, class_score, n_samples[class_name]]
                for name, score in calculate_metrics(list(scorers.values()), dataset, model,
                                                     prediction_extract=self.prediction_extract).items()
                # scorer returns numpy array of results with item per class
                for class_score, class_name in zip(score, classes)
            )
        results = []
        # We put a NoOp for first spot (e.g. identity) so we can replace first Op at every iteration
        augmented_dataset.get_data_loader().dataset.transform = A.Compose(
            [A.NoOp()] + augmented_dataset.dataset.transform.transforms.transforms)

        for a in self.augmentations:
            # We will override the first augmentation, the one that is currently identity,
            # with the one we want to test for robustness
            curr_img_transform = augmented_dataset.get_data_loader().dataset.trasform
            augmented_dataset.get_data_loader().dataset.transform = A.Compose([a] + curr_img_transform.transforms[1:])
            results_base = evaluate_dataset(baseline_dataset)
            results_aug = evaluate_dataset(augmented_dataset)


        results_df = pd.DataFrame(results, columns=['Class', 'Metric', 'Value', 'Number of samples']
                                  ).sort_values(by=['Class'])
        return results_df
