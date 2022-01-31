"""Module containing robustness report check."""
import numpy as np
import random
import torch
from typing import Callable, TypeVar, List, Optional, Union
import albumentations as A
import pandas as pd
import plotly.express as px
from ignite.metrics import Metric
import torch.utils.data as data

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
                 augmentations: Optional[Union[A.Compose, List[A.BasicTransform]]] = None,
                 epsilon: float = 10 ** -2):
        super().__init__()
        self._epsilon = epsilon
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

    def set_seeds(self, seed: int = 42):
        """
        Sets seeds for reproduceability
        Imgaug uses numpy's State
        Albumentation uses Python and imgaug seeds
        :param seed:
        :return:
        """
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

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
        # This takes time, consider replacing
        if set(baseline_dataset.get_samples_per_class().keys()) != set(augmented_dataset.get_samples_per_class().keys()):
            raise DeepchecksValueError("Datasets must share class count")
        if not all([baseline_dataset.get_samples_per_class()[k] == augmented_dataset.get_samples_per_class()[k]
                    for k in baseline_dataset.get_samples_per_class().keys()]):
            raise DeepchecksValueError("Dataset must have same numnber of examples per class")
        baseline_dataset.validate_transforms()
        augmented_dataset.validate_transforms()
        baseline_dataset.validate_label()
        augmented_dataset.validate_label()
        baseline_dataset.validate_shared_label(augmented_dataset)
        model_type_validation(model)

        task_type = task_type_check(model, baseline_dataset)

        # Get default scorers if no alternative, or validate alternatives
        scorers = get_scorers_list(model, baseline_dataset, baseline_dataset.get_num_classes(),
                                   self.alternative_metrics)
        if task_type in (TaskType.CLASSIFICATION, TaskType.OBJECT_DETECTION):
            results_df = self._report_inner_loop(baseline_dataset, augmented_dataset, model, scorers)
        else:
            return NotImplementedError('only works for classification ATM')

        example_dict = self._get_bad_aug_examples(baseline_dataset, augmented_dataset, results_df)
        bad_example_figures = self._create_example_figure(example_dict)

        # TODO visualiztion
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
        # We put a NoOp for first spot (e.g. identity) so we can replace first Op at every iteration
        augmented_dataset.add_dataset_transforms()
        results = []
        # Run baseline
        curr_results_base = evaluate_dataset(baseline_dataset)
        results_base_df = pd.DataFrame(curr_results_base, columns=['Class', 'Metric', 'Value', 'N']
                                       ).sort_values(by=['Class'])
        for curr_aug in self.augmentations:
            aug_name = curr_aug.get_class_fullname()
            # We will override the first augmentation, the one that is currently identity,
            # with the one we want to test for robustness
            augmented_dataset.edit_dataset_transforms(curr_aug)
            curr_results_aug = evaluate_dataset(augmented_dataset)
            results_aug_df = pd.DataFrame(curr_results_aug, columns=['Class', 'Metric', 'Value', 'N']
                                          ).sort_values(by=['Class'])
            # This modifies the augmentation name in-place
            results_base_df["Augmentation"] = aug_name
            # This adds the augmentation
            results_aug_df["Augmentation"] = aug_name
            results.append(pd.concat([results_base_df, results_aug_df], keys=["Baseline", "Augmented"]))

        # Create grand DataFrame from dictionary of augmentations
        # This has flat hierarchy for augmentations and baseline-vs-augmentations
        results_df = pd.concat(results, axis=0).\
            reset_index().\
            rename({"level_0": "Status"}, axis=1).\
            drop(labels="level_1", axis=1)

        def _is_robust_to_augmentation(p, epsilon=10 ** -2):
            grouped_results = p[["Status", "Metric", "Value"]].groupby(["Status", "Metric"]).median().reset_index()
            differences = []
            metric_statuses = []
            for metric in grouped_results["Metric"].unique():
                diff = grouped_results[grouped_results["Metric"] == metric]["Value"].diff().abs().iloc[-1]
                differences.append(diff > epsilon)
                metric_statuses.append(diff)
            difference = any(differences)
            return pd.Series([difference] + metric_statuses, index=["Affected"] +
                                                                   [f"{m}_delta" for m in
                                                                    grouped_results["Metric"].unique().tolist()])

        # Iterating this dataframe per [Status, Augmentation] will give us easy comparisons
        metric_results = results_df.groupby(["Augmentation"]).apply(_is_robust_to_augmentation, self._epsilon).reset_index()

        # TODO turn result list into a plot
        return metric_results

    def _get_random_image_pairs_from_dataloder(self, baseline_dataset, augmented_dataset, n_samples=10):
        """
        We iterate the internal dataset object directly to avoid randomness
        Dataset returns data points as processed images, making this currently not really usable
        To avoid making more assumptions this currently stays as-is
        Note that images return in RGB format, ond to visualize them using OpenCV the final dimesion should be
        transposed;
        can be done via image = image[:, :, ::-1] or cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        :param baseline_dataset:
        :param augmented_dataset:
        :param n_samples:
        :return:
        """
        # We definitely make the assumption that the underlying structure is torch.utils.Dataset
        baseline_sampler = iter(baseline_dataset.get_data_loader().dataset)
        aug_sampler = iter(augmented_dataset.get_data_loader().dataset)
        samples = []
        # iterate and sample
        for idx, (sample_base, sample_aug) in enumerate(zip(baseline_sampler, aug_sampler)):
            if idx > n_samples:
                break
            # TODO this method uses
            sample_base_ = baseline_dataset.inverse_transform(sample_base[0])
            sample_aug_ = augmented_dataset.inverse_transform(sample_aug[0])
            samples.append((sample_base_, sample_aug_))

        return samples

    def _get_bad_aug_examples(self, baseline_dataset, augmented_dataset, results_df):
        bad_aug_names = results_df[results_df["Affected"]]["Augmentation"].tolist()
        bad_augs = [a for a in self.augmentations if a.get_class_fullname() in bad_aug_names]
        sample_dict = {}
        for aug in bad_augs:
            augmented_dataset.edit_dataset_transforms(aug)
            sample_dict[aug.get_class_fullname()] = self._get_random_image_pairs_from_dataloder(baseline_dataset,
                                                                                                augmented_dataset)
        return sample_dict

    def _create_example_figure(self, example_dict):
        # TODO not fully implemented
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        figures = []
        for aug_name, aug_images in example_dict.items():
            fig = make_subplots(rows=3, cols=3)
            for i in range(3):
                for j in range(3):
                    print(j + (i * 3))
                    fig.add_trace(
                        go.Image(z=aug_images[j + (i * 3)]), row=i+1, col=j+1
                    )
            fig.update_layout(height=600, width=800)
            figures.append(fig)
        return figures
