# ----------------------------------------------------------------------------
# Copyright (C) 2021-2022 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""Module containing robustness report check."""
import string
from collections import defaultdict
from random import choice
from typing import Dict, List, Optional, Sequence, Sized, TypeVar

import albumentations
import imgaug
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from ignite.metrics import Metric
from plotly.subplots import make_subplots

from deepchecks.core import CheckResult, ConditionResult
from deepchecks.core.condition import ConditionCategory
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.utils import plot
from deepchecks.utils.strings import format_percent, split_camel_case
from deepchecks.vision import Batch, Context, SingleDatasetCheck, VisionData
from deepchecks.vision.metrics_utils import calculate_metrics, get_scorers_dict, metric_results_to_df
from deepchecks.vision.utils.image_functions import ImageInfo, draw_bboxes, prepare_thumbnail
from deepchecks.vision.utils.transformations import AbstractTransformations
from deepchecks.vision.utils.validation import set_seeds
from deepchecks.vision.vision_data import TaskType

__all__ = ['RobustnessReport']


PR = TypeVar('PR', bound='RobustnessReport')


class RobustnessReport(SingleDatasetCheck):
    """Compare performance of model on original dataset and augmented dataset.

    Parameters
    ----------
    alternative_metrics : Dict[str, Metric], default: None
        A dictionary of metrics, where the key is the metric name and the value is an ignite.Metric object whose score
        should be used. If None are given, use the default metrics.
    augmentations : List, default: None
        A list of augmentations to test on the data. If none are given default augmentations are used.
        Supported augmentations are of albumentations and imgaug.
    """

    _THUMBNAIL_SIZE = (150, 150)

    def __init__(self,
                 alternative_metrics: Optional[Dict[str, Metric]] = None,
                 augmentations: List = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.alternative_metrics = alternative_metrics
        self.augmentations = augmentations

        self._state = None

    def initialize_run(self, context: Context, dataset_kind):
        """Initialize the metrics for the check, and validate task type is relevant."""
        dataset = context.get_data_by_kind(dataset_kind)
        # Set empty version of metrics
        self._state = {'metrics': get_scorers_dict(dataset, self.alternative_metrics)}

    def update(self, context: Context, batch: Batch, dataset_kind):
        """Accumulates batch data into the metrics."""
        label = batch.labels
        prediction = batch.predictions
        for _, metric in self._state['metrics'].items():
            metric.update((prediction, label))

    def compute(self, context: Context, dataset_kind) -> CheckResult:
        """Run check.

        Returns
        -------
            CheckResult: value is dictionary in format 'score-name': score-value
        """
        dataset = context.get_data_by_kind(dataset_kind)
        model = context.model

        # Validate the transformations works
        transforms_handler = dataset.get_transform_type()
        self._validate_augmenting_affects(transforms_handler, dataset)
        # Return dataframe of (Class, Metric, Value)
        base_results: pd.DataFrame = metric_results_to_df(
            {k: m.compute() for k, m in self._state['metrics'].items()}, dataset
        )
        # TODO: update later the way we handle average metrics

        # Return dict of metric to value
        base_mean_results: dict = self._calc_mean_metrics(base_results)
        # Get augmentations
        augmentations = self.augmentations or transforms_handler.get_robustness_augmentations(dataset.data_dimension)
        aug_all_data = {}
        for augmentation_func in augmentations:
            aug_dataset = dataset.get_augmented_dataset(augmentation_func)
            # The metrics have saved state, but they are reset inside `calculate_metrics`
            metrics = self._state['metrics']
            # The augmentations are pseudo-random and affected by the seeds.
            # Setting it here to have fixed state just before the augmentations are run
            set_seeds(context.random_state)
            # Return dataframe of (Class, Metric, Value)
            aug_results = metric_results_to_df(
                calculate_metrics(metrics, aug_dataset, model, context.device),
                aug_dataset
            )
            # Return dict of {metric: {'score': mean score, 'diff': diff from base}, ... }
            metrics_diff_dict = self._calc_performance_diff(base_mean_results, aug_results)
            # Return dict of metric to list {metric: [{'class': x, 'value': y, 'diff': z, 'samples': w}, ...], ...}
            top_affected_classes = self._calc_top_affected_classes(base_results, aug_results, dataset, 5)
            # Return list of [(base image, augmented image, class, [bbox,...]), ...]
            image_pairs = get_random_image_pairs_from_dataset(dataset, aug_dataset, top_affected_classes)
            aug_all_data[augmentation_name(augmentation_func)] = {
                'metrics': aug_results,
                'metrics_diff': metrics_diff_dict,
                'top_affected': top_affected_classes,
                'images': image_pairs
            }

        # Save as result only the metrics diff per augmentation
        result = {aug: data['metrics_diff'] for aug, data in aug_all_data.items()}

        if context.with_display:
            # Create figures to display
            aug_names = ', '.join([augmentation_name(aug) for aug in augmentations])
            info_message = 'Percentage shown are difference between the metric before augmentation and after.<br>' \
                f'Augmentations used (separately): {aug_names}'
            figures = self._create_augmentation_figures(dataset, base_mean_results, aug_all_data)
            display = [info_message, *figures]
        else:
            display = None

        return CheckResult(
            result,
            header='Robustness Report',
            display=display
        )

    def add_condition_degradation_not_greater_than(self, ratio: float = 0.02):
        """Add condition which validates augmentations doesn't degrade the model metrics by given amount."""
        def condition(result):
            failed = [
                aug
                for aug, metrics in result.items()
                for _, metric_data in metrics.items()
                if metric_data['diff'] < -1 * ratio
            ]

            if not failed:
                return ConditionResult(ConditionCategory.PASS)
            else:
                details = f'Augmentations not passing: {set(failed)}'
                return ConditionResult(ConditionCategory.FAIL, details)

        return self.add_condition(f'Metrics degrade by not more than {format_percent(ratio)}', condition)

    def _validate_augmenting_affects(self, transform_handler: AbstractTransformations, dataset: VisionData):
        """Validate the user is using the transforms' field correctly, and that if affects the image and label."""
        aug_dataset = dataset.get_augmented_dataset(transform_handler.get_test_transformation())
        # Iterate both datasets and compare results
        baseline_sampler = iter(dataset.data_loader.dataset)
        aug_sampler = iter(aug_dataset.data_loader.dataset)

        # Validating on a single sample that the augmentation had affected
        for (sample_base, sample_aug) in zip(baseline_sampler, aug_sampler):
            # Skips any sample without label
            label = sample_base[1]
            if label is None or (isinstance(label, Sized) and len(label) == 0):
                continue

            batch = dataset.to_batch(sample_base, sample_aug)
            images = dataset.batch_to_images(batch)
            if ImageInfo(images[0]).is_equals(images[1]):
                msg = f'Found that images have not been affected by adding augmentation to field ' \
                      f'"{dataset.transform_field}". This might be a problem with the implementation of ' \
                      f'Dataset.__getitem__'
                raise DeepchecksValueError(msg)

            # For object detection check that the label is affected
            if dataset.task_type == TaskType.OBJECT_DETECTION and transform_handler.is_transforming_labels:
                labels = dataset.batch_to_labels(batch)
                if torch.equal(labels[0], labels[1]):
                    msg = f'Found that labels have not been affected by adding augmentation to field ' \
                          f'"{dataset.transform_field}". This might be a problem with the implementation of ' \
                          f'`Dataset.__getitem__`. label value: {labels[0]}'
                    raise DeepchecksValueError(msg)
            # If all validations passed return
            return

    def _calc_top_affected_classes(self, base_results, augmented_results, dataset, n_classes_to_show):
        def calc_percent(a, b):
            return (a - b) / b if b != 0 else 0

        aug_top_affected = defaultdict(list)
        metrics = base_results['Metric'].unique().tolist()
        for metric in metrics:
            single_metric_scores = augmented_results[augmented_results['Metric'] == metric][['Class', 'Value']] \
                .set_index('Class')
            single_metric_scores['Base'] = base_results[base_results['Metric'] == metric][['Class', 'Value']] \
                .set_index('Class')
            diff = single_metric_scores.apply(lambda x: calc_percent(x.Value, x.Base), axis=1)

            for index_class, diff_value in diff.sort_values().iloc[:n_classes_to_show].iteritems():
                aug_top_affected[metric].append({'class': index_class,
                                                 'value': single_metric_scores.at[index_class, 'Value'],
                                                 'diff': diff_value,
                                                 'samples': dataset.n_of_samples_per_class.get(index_class, 0)})
        return aug_top_affected

    def _calc_performance_diff(self, mean_base, augmented_metrics):
        def difference(aug_score, base_score):
            if base_score == 0:
                # If base score is 0 can't divide by it, so if aug score equals returns 0 which means no difference,
                # else return negative or positive infinity depends on direction of aug
                return 0 if aug_score == 0 else np.inf if aug_score > base_score else -np.inf
            return (aug_score - base_score) / base_score

        diff_dict = {}
        for metric, score in self._calc_mean_metrics(augmented_metrics).items():
            diff_dict[metric] = {'score': score, 'diff': difference(score, mean_base[metric])}

        return diff_dict

    def _calc_mean_metrics(self, metrics_df: pd.DataFrame) -> dict:
        metrics_df = metrics_df[['Metric', 'Value']].groupby(['Metric']).aggregate('mean')
        return metrics_df.to_dict()['Value']

    def _create_augmentation_figures(self, dataset, base_mean_results, aug_all_data):
        figures = []

        def sort_by_worst_func(aug_data):
            return sum([m['score'] for m in aug_data[1]['metrics_diff'].values()])

        sorted_by_worst = dict(sorted(aug_all_data.items(), key=sort_by_worst_func))

        # Iterate augmentations
        for index, (augmentation, curr_data) in enumerate(sorted_by_worst.items()):
            # Create example figures, return first n_pictures_to_show from original and then n_pictures_to_show from
            # augmented dataset
            figures.append(self._create_example_figure(dataset, curr_data['images'], augmentation))
            # Create performance graph
            figures.append(self._create_performance_graph(base_mean_results, curr_data['metrics_diff']))
            # Create top affected graph
            figures.append(self._create_top_affected_graph(curr_data['top_affected'], dataset))
            if index < len(aug_all_data) - 1:
                figures.append('<hr style="background-color:#2a3f5f; height:5px">')
        return figures

    def _create_example_figure(self, dataset: VisionData, images, aug_name: str):
        classes = []
        base_images = []
        aug_images = []

        for sample in images:
            base_image = sample[0]
            aug_image = sample[1]
            class_name = dataset.label_id_to_name(sample[2])
            classes.append(f'<h4>{class_name}</h4>')

            if len(sample) == 4:
                base_image_bboxes, aug_image_bboxes = sample[3]
                base_image = draw_bboxes(base_image, base_image_bboxes, copy_image=True, border_width=2)
                aug_image = draw_bboxes(aug_image, aug_image_bboxes, copy_image=True, border_width=2)

            base_images.append(prepare_thumbnail(
                image=base_image,
                size=self._THUMBNAIL_SIZE,
                copy_image=False
            ))
            aug_images.append(prepare_thumbnail(
                image=aug_image,
                size=self._THUMBNAIL_SIZE,
                copy_image=False
            ))

        # Create id of alphabetic characters
        sid = ''.join([choice(string.ascii_uppercase) for _ in range(6)])
        classes = ''.join([f'<div class="{sid}-item">{x}</div>' for x in classes])
        base_images_thumbnails = ''.join([f'<div class="{sid}-item">{x}</div>' for x in base_images])
        aug_images_thumbnails = ''.join([f'<div class="{sid}-item">{x}</div>' for x in aug_images])

        return HTML_TEMPLATE.format(
            aug_name=aug_name,
            classes=classes,
            base_images=base_images_thumbnails,
            aug_images=aug_images_thumbnails,
            id=sid
        )

    def _create_performance_graph(self, base_scores: dict, augmented_scores: dict):
        metrics = sorted(list(base_scores.keys()))
        fig = make_subplots(rows=1, cols=len(metrics), subplot_titles=metrics)

        for index, metric in enumerate(metrics):
            curr_aug = augmented_scores[metric]
            x = ['Origin', 'Augmented']
            y = [base_scores[metric], curr_aug['score']]
            diff = ['', format_percent(curr_aug['diff'])]

            fig.add_trace(go.Bar(x=x, y=y, customdata=diff, texttemplate='%{customdata}',
                                 textposition='auto', marker=dict(color=plot.metric_colors[index])),
                          col=index + 1, row=1)

        return fig \
            .update_layout(
                font=dict(size=12),
                height=300,
                autosize=False,
                title=dict(
                    text='Performance Comparison',
                    font=dict(size=20)),
                margin=dict(l=0, b=0),
                showlegend=False
            ).update_xaxes(
                title=None,
                type='category',
                tickangle=30,
                # NOTE:
                # the range, in this case, is needed to fix a problem with
                # too wide bars when there are only one or two of them`s on
                # the plot, plus it also centralizes them`s on the plot
                # The min value of the range (range(min. max)) is bigger because
                # otherwise bars will not be centralized on the plot, they will
                # appear on the left part of the plot (that is probably because of zero)
                range=(-2, 3)
            )

    def _create_top_affected_graph(self, top_affected_dict, dataset):
        metrics = sorted(top_affected_dict.keys())
        fig = make_subplots(rows=1, cols=len(metrics), subplot_titles=metrics)
        max_n_of_classes = 0

        for index, metric in enumerate(metrics):
            metric_classes = top_affected_dict[metric]

            # Take the worst affected classes
            x = []
            y = []
            custom_data = []

            for class_info in metric_classes:
                x.append(dataset.label_id_to_name(class_info['class']))
                y.append(class_info['value'])
                custom_data.append([format_percent(class_info['diff']), class_info['samples']])

            if len(x) > max_n_of_classes:
                max_n_of_classes = len(x)

            # Plotly have a bug that if all y values are zero text position 'auto' doesn't work
            textposition = 'outside' if sum(y) == 0 else 'auto'
            hover = 'Metric value: %{y:.2f}<br>Number of samples: %{customdata[1]}'
            fig.add_trace(go.Bar(name=metric, x=x, y=y, customdata=custom_data, texttemplate='%{customdata[0]}',
                                 textposition=textposition, hovertemplate=hover,
                                 marker=dict(color=plot.metric_colors[index])),
                          row=1, col=index + 1)

        return fig \
            .update_layout(
                font=dict(size=12),
                height=300,
                title=dict(
                    text='Top Affected Classes',
                    font=dict(size=20)),
                margin=dict(l=0, b=0),
                showlegend=False
            ).update_xaxes(
                title=None,
                type='category',
                tickangle=30,
                tickprefix='Class ',
                automargin=True,
                # NOTE:
                # the range, in this case, is needed to fix a problem with
                # too wide bars when there are only one or two of them`s on
                # the plot, plus it also centralizes them`s on the plot
                # The min value of the range (range(min. max)) is bigger because
                # otherwise bars will not be centralized on the plot, they will
                # appear on the left part of the plot (that is probably because of zero)
                range=(-2, max_n_of_classes + 1)
            ).update_yaxes(
                automargin=True
            )


def augmentation_name(aug):
    if isinstance(aug, imgaug.augmenters.Augmenter):
        name = aug.name
    elif isinstance(aug, albumentations.BasicTransform):
        name = aug.get_class_fullname()
    elif isinstance(aug, torch.nn.Module):
        name = aug.__class__.__name__
    else:
        raise DeepchecksValueError(f'Unsupported augmentation type {type(aug)}')

    return split_camel_case(name)


def get_random_image_pairs_from_dataset(original_dataset: VisionData,
                                        augmented_dataset: VisionData,
                                        top_affected_classes: dict):
    """Get image pairs from 2 datasets."""
    classes_to_show = {
        class_info['class']: class_info['diff']
        for classes_list in top_affected_classes.values()
        for class_info in classes_list
    }

    # Sorting classes by diff value
    classes = [k for k, v in sorted(classes_to_show.items(), key=lambda item: item[1])]
    samples = []

    for class_id in classes:
        # Takes the dataset index of a sample of the given class. The order in the dataset is equal for both original
        # and augmented dataset, so can use it on both
        dataset_class_index = original_dataset.classes_indices[class_id][0]

        sample_base = original_dataset.data_loader.dataset[dataset_class_index]
        sample_aug = augmented_dataset.data_loader.dataset[dataset_class_index]
        batch = original_dataset.to_batch(sample_base, sample_aug)
        images: Sequence[np.ndarray] = original_dataset.batch_to_images(batch)

        if original_dataset.task_type == TaskType.OBJECT_DETECTION:
            batch_label: torch.Tensor = original_dataset.batch_to_labels(batch)
            base_label: torch.Tensor = batch_label[0]
            aug_label: torch.Tensor = batch_label[1]
            # Take only bboxes of this class
            base_class_label = [x for x in base_label if x[0] == class_id]
            aug_class_label = [x for x in aug_label if x[0] == class_id]
            samples.append((images[0], images[1], class_id, (base_class_label, aug_class_label)))
        else:
            samples.append((images[0], images[1], class_id))

    return samples


HTML_TEMPLATE = """
<style>
    .{id}-container {{
        overflow-x: auto;
        display: flex;
        flex-direction: column;
        gap: 10px;
    }}
    .{id}-row {{
      display: flex;
      flex-direction: row;
      align-items: center;
      gap: 10px;
    }}
    .{id}-item {{
      display: flex;
      min-width: 200px;
      position: relative;
      word-wrap: break-word;
      align-items: center;
      justify-content: center;
    }}
    .{id}-title {{
        font-family: "Open Sans", verdana, arial, sans-serif;
        color: #2a3f5f
    }}
    /* A fix for jupyter widget which doesn't have width defined on HTML widget */
    .widget-html-content {{
        width: -moz-available;          /* WebKit-based browsers will ignore this. */
        width: -webkit-fill-available;  /* Mozilla-based browsers will ignore this. */
        width: fill-available;
    }}
</style>
<h3><b>Augmentation "{aug_name}"</b></h3>
<div class="{id}-container">
    <div class="{id}-row">
        <h5 class="{id}-item">Class</h5>
        {classes}
    </div>
    <div class="{id}-row">
        <h5 class="{id}-item">Base Image</h5>
        {base_images}
    </div>
    <div class="{id}-row">
        <h5 class="{id}-item">Augmented Image</h5>
        {aug_images}
    </div>
</div>
"""
