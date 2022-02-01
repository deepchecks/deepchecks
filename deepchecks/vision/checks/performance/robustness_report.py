"""Module containing robustness report check."""
from copy import copy

import numpy as np
import random
import torch
import imgaug
from typing import Callable, TypeVar, List, Optional
import albumentations

import pandas as pd
import plotly.express as px
from ignite.metrics import Metric

from deepchecks import CheckResult
from deepchecks.vision import VisionDataset, SingleDatasetCheck, Context
from deepchecks.vision.dataset import TaskType
from deepchecks.vision.metrics_utils import calculate_metrics

__all__ = ['RobustnessReport']

from vision.utils.transformations import add_augmentation

PR = TypeVar('PR', bound='RobustnessReport')


class RobustnessReport(SingleDatasetCheck):
    """Check several image enhancements for model robustness.

    Parameters
    ----------
        alternative_metrics (List[Metric], default None):
            A list of ignite.Metric objects whose score should be used. If None are given, use the default metrics.
    """

    def __init__(self,
                 alternative_metrics: Optional[List[Metric]] = None,
                 prediction_extract: Optional[Callable] = None,
                 augmentations=None,
                 epsilon: float = 10 ** -2,
                 random_seed: int = 42):
        super().__init__()
        self._epsilon = epsilon
        self.alternative_metrics = alternative_metrics
        self.prediction_extract = prediction_extract
        self.random_seed = random_seed
        self.augmentations = augmentations

    def run_logic(self, context: Context, dataset_type: str = 'train') -> CheckResult:
        """Run check.

        Returns
        -------
            CheckResult: value is dictionary in format 'score-name': score-value
        """
        self._set_seeds(self.random_seed)
        context.assert_task_type(TaskType.CLASSIFICATION, TaskType.OBJECT_DETECTION)
        if dataset_type == 'train':
            dataset = context.train
        else:
            dataset = context.test

        transform_type = dataset.transform_type
        task_type = dataset.task_type
        model = context.model
        self.augmentations = self.augmentations or get_robustness_augmentations(transform_type)
        # Get default scorers if no alternative, or validate alternatives
        # scorers = get_scorers_list(model, dataset, dataset.get_num_classes(), self.alternative_metrics)
        scorers = []

        results_df, examples = self._report_inner_loop(dataset, model, scorers)

        bad_example_figures = self._create_example_figure(examples)

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

    def _report_inner_loop(self, dataset: VisionDataset, model, scorers):
        n_samples = dataset.get_samples_per_class()
        classes = n_samples.keys()

        def evaluate(data_loader):
            return (
                [class_name, name, class_score, n_samples[class_name]]
                for name, score in calculate_metrics(list(scorers.values()), dataset, data_loader, model,
                                                     prediction_extract=self.prediction_extract).items()
                # scorer returns numpy array of results with item per class
                for class_score, class_name in zip(score, classes)
            )

        # Run baseline
        curr_results_base = evaluate(dataset.get_data_loader())
        results_base_df = pd.DataFrame(curr_results_base, columns=['Class', 'Metric', 'Value', 'N']
                                       ).sort_values(by=['Class'])

        results = []
        examples = []
        for curr_aug in self.augmentations:
            aug_name = augmentation_name(curr_aug)
            # Create a copy of data loader and the dataset
            copy_data_loader = copy(dataset.get_data_loader())
            copy_data_loader.dataset = copy(copy_data_loader.dataset)
            # Add augmentation in the first place
            add_augmentation(copy_data_loader.dataset, dataset.transform_field, curr_aug)
            # Evaluate
            curr_results_aug = evaluate(copy_data_loader)
            results_aug_df = pd.DataFrame(curr_results_aug, columns=['Class', 'Metric', 'Value', 'N']
                                          ).sort_values(by=['Class'])
            # This modifies the augmentation name in-place
            results_base_df["Augmentation"] = aug_name
            # This adds the augmentation
            results_aug_df["Augmentation"] = aug_name
            results.append(pd.concat([results_base_df, results_aug_df], keys=["Baseline", "Augmented"]))

            examples.append(self._get_random_image_pairs_from_dataloder(dataset.get_data_loader(), copy_data_loader))

        # Create grand DataFrame from dictionary of augmentations
        # This has flat hierarchy for augmentations and baseline-vs-augmentations
        results_df = pd.concat(results, axis=0). \
            reset_index(). \
            rename({"level_0": "Status"}, axis=1). \
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
        metric_results = results_df.groupby(["Augmentation"]).apply(_is_robust_to_augmentation,
                                                                    self._epsilon).reset_index()

        return metric_results, examples

    def _get_random_image_pairs_from_dataloder(self, baseline_data_loader, augmented_data_loader, n_samples=10):
        """
        We iterate the internal dataset object directly to avoid randomness
        Dataset returns data points as processed images, making this currently not really usable
        To avoid making more assumptions this currently stays as-is
        Note that images return in RGB format, ond to visualize them using OpenCV the final dimesion should be
        transposed;
        can be done via image = image[:, :, ::-1] or cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        """
        # We definitely make the assumption that the underlying structure is torch.utils.Dataset
        baseline_sampler = iter(baseline_data_loader.dataset)
        aug_sampler = iter(augmented_data_loader.dataset)
        samples = []
        # iterate and sample
        for idx, (sample_base, sample_aug) in enumerate(zip(baseline_sampler, aug_sampler)):
            if idx > n_samples:
                break
            samples.append((sample_base[0], sample_aug[0]))

        return samples

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
                        go.Image(z=aug_images[j + (i * 3)]), row=i + 1, col=j + 1
                    )
            fig.update_layout(height=600, width=800)
            figures.append(fig)
        return figures

    def _set_seeds(self, seed):
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


def get_robustness_augmentations(transform_type):
    if transform_type == 'albumentations':
        # Note that p=1.0 since we want to apply those to entire dataset
        return [
            albumentations.RandomBrightnessContrast(p=1.0),
            albumentations.ShiftScaleRotate(p=1.0),
            albumentations.HueSaturationValue(p=1.0),
            albumentations.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=1.0),
        ]
    # imgaug augmentations works also inside pytorch compose
    else:
        return [
            imgaug.augmenters.MultiplyHueAndSaturation()
        ]


def augmentation_name(aug):
    if isinstance(aug, imgaug.augmenters.Augmenter):
        return aug.name
    elif isinstance(aug, albumentations.BasicTransform):
        return aug.get_class_fullname()
    else:
        return type(aug).__name__
