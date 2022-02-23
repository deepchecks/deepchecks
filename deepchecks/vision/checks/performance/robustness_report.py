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
"""Module containing robustness report check."""
from collections import defaultdict

import imgaug
from typing import TypeVar, List, Optional, Any, Sized, Dict
import albumentations
import numpy as np

import pandas as pd
import torch
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from ignite.metrics import Metric

from deepchecks import CheckResult, ConditionResult
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.vision import VisionData, SingleDatasetCheck, Context
from deepchecks.vision.dataset import TaskType
from deepchecks.vision.metrics_utils import calculate_metrics, metric_results_to_df
from deepchecks.vision.utils.validation import set_seeds
from deepchecks.vision.metrics_utils import get_scorers_list
from deepchecks.utils.strings import format_percent
from deepchecks.vision.utils.image_functions import numpy_to_image_figure, apply_heatmap_image_properties, \
    label_bbox_add_to_figure, ImageInfo


__all__ = ['RobustnessReport']


PR = TypeVar('PR', bound='RobustnessReport')


class RobustnessReport(SingleDatasetCheck):
    """Check several image enhancements for model robustness.

    Parameters
    ----------
    alternative_metrics : Dict[str, Metric], default: None
        A dictionary of metrics, where the key is the metric name and the value is an ignite.Metric object whose score
        should be used. If None are given, use the default metrics.
    augmentations : List, default: None
        A list of augmentations to test on the data. If none are given default augmentations are used.
        Supported augmentations are of albumentations and imgaug.
    """

    def __init__(self,
                 alternative_metrics: Optional[Dict[str, Metric]] = None,
                 augmentations: List = None):
        super().__init__()
        self.alternative_metrics = alternative_metrics
        self.augmentations = augmentations
        self._state = None

    def initialize_run(self, context: Context, dataset_kind):
        """Initialize the metrics for the check, and validate task type is relevant."""
        context.assert_task_type(TaskType.CLASSIFICATION, TaskType.OBJECT_DETECTION)
        dataset = context.get_data_by_kind(dataset_kind)
        # Set empty version of metrics
        self._state = {'metrics': get_scorers_list(dataset, self.alternative_metrics)}

    def update(self, context: Context, batch: Any, dataset_kind):
        """Accumulates batch data into the metrics."""
        dataset = context.get_data_by_kind(dataset_kind)
        label = dataset.label_formatter(batch)
        # Using context.infer to get cached prediction if exists
        prediction = context.infer(batch)
        for _, metric in self._state['metrics'].items():
            metric.update((prediction, label))

    def compute(self, context: Context, dataset_kind) -> CheckResult:
        """Run check.

        Returns
        -------
            CheckResult: value is dictionary in format 'score-name': score-value
        """
        set_seeds(context.random_state)
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
        base_mean_results: dict = self._calc_median_metrics(base_results)
        # Get augmentations
        augmentations = self.augmentations or transforms_handler.get_robustness_augmentations(dataset.data_dimension)
        aug_all_data = {}
        for augmentation_func in augmentations:
            augmentation = augmentation_name(augmentation_func)
            aug_dataset = self._create_augmented_dataset(dataset, augmentation_func)
            # The metrics have saved state, but they are reset inside `calculate_metrics`
            metrics = self._state['metrics']
            # Return dataframe of (Class, Metric, Value)
            aug_results = metric_results_to_df(
                calculate_metrics(metrics, aug_dataset, model, context.prediction_formatter), aug_dataset
            )
            # Return dict of {metric: {'score': mean score, 'diff': diff from base}, ... }
            metrics_diff_dict = self._calc_performance_diff(base_mean_results, aug_results)
            # Return dict of metric to list {metric: [{'class': x, 'value': y, 'diff': z, 'samples': w}, ...], ...}
            top_affected_classes = self._calc_top_affected_classes(base_results, aug_results, dataset, 5)
            # Return list of [(base image, augmented image, class), ...]
            image_pairs = get_random_image_pairs_from_dataset(dataset, aug_dataset, top_affected_classes)
            aug_all_data[augmentation] = {
                'metrics': aug_results,
                'metrics_diff': metrics_diff_dict,
                'top_affected': top_affected_classes,
                'images': image_pairs
            }

        # Create figures to display
        figures = self._create_augmentation_figure(dataset, base_mean_results, aug_all_data)

        # Save as result only the metrics diff per augmentation
        result = {aug: data['metrics_diff'] for aug, data in aug_all_data.items()}

        return CheckResult(
            result,
            header='Robustness Report',
            display=figures
        )

    def add_condition_degradation_not_greater_than(self, ratio: 0.01):
        """Add condition which validates augmentations doesn't degrade the model metrics by given amount."""
        def condition(result):
            failed = [
                aug
                for aug, metrics in result.items()
                for _, metric_data in metrics.items()
                if metric_data['diff'] < -1 * ratio
            ]

            if not failed:
                return ConditionResult(True)
            else:
                details = f'Augmentations not passing: {set(failed)}'
                return ConditionResult(False, details)

        return self.add_condition(f'Metrics degrade by not more than {format_percent(ratio)}', condition)

    def _create_augmented_dataset(self, dataset: VisionData, augmentation_func):
        # Create a copy of data loader and the dataset
        aug_dataset: VisionData = dataset.copy()
        # Add augmentation in the first place
        aug_dataset.add_augmentation(augmentation_func)
        return aug_dataset

    def _validate_augmenting_affects(self, transform_handler, dataset: VisionData):
        """Validate the user is using the transforms' field correctly, and that if affects the image and label."""
        aug_dataset = self._create_augmented_dataset(dataset, transform_handler.get_test_transformation())
        # Iterate both datasets and compare results
        baseline_sampler = iter(dataset.get_data_loader().dataset)
        aug_sampler = iter(aug_dataset.get_data_loader().dataset)

        # Validating on a single sample that the augmentation had affected
        for (sample_base, sample_aug) in zip(baseline_sampler, aug_sampler):
            # Skips any sample without label
            label = sample_base[1]
            if label is None or (isinstance(label, Sized) and len(label) == 0):
                continue

            batch = dataset.to_batch(sample_base, sample_aug)
            images = dataset.image_formatter(batch)
            if ImageInfo(images[0]).is_equals(images[1]):
                msg = f'Found that images have not been affected by adding augmentation to field ' \
                      f'"{dataset.transform_field}". This might be a problem with the implementation of ' \
                      f'Dataset.__getitem__'
                raise DeepchecksValueError(msg)

            # For classification does not check label for difference
            if dataset.task_type != TaskType.CLASSIFICATION:
                labels = dataset.label_formatter(batch)
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
                                                 'samples': dataset.n_of_samples_per_class[index_class]})
        return aug_top_affected

    def _calc_performance_diff(self, mean_base, augmented_metrics):
        def difference(aug_score, base_score):
            if base_score == 0:
                # If base score is 0 can't divide by it, so if aug score equals returns 0 which means no difference,
                # else return negative or positive infinity depends on direction of aug
                return 0 if aug_score == 0 else np.inf if aug_score > base_score else -np.inf
            return (aug_score - base_score) / base_score

        diff_dict = {}
        for metric, score in self._calc_median_metrics(augmented_metrics).items():
            diff_dict[metric] = {'score': score, 'diff': difference(score, mean_base[metric])}

        return diff_dict

    def _calc_median_metrics(self, metrics_df) -> dict:
        metrics_df = metrics_df[['Metric', 'Value']].groupby(['Metric']).median()
        return metrics_df.to_dict()['Value']

    def _create_augmentation_figure(self, dataset, base_mean_results, aug_all_data):
        figures = []
        # Iterate augmentation names
        for augmentation, curr_data in aug_all_data.items():
            figures.append(f'<h3>Augmentation: {augmentation}</h3>')
            # Create performance graph
            figures.append(self._create_performance_graph(base_mean_results, curr_data['metrics_diff']))
            # Create top affected graph
            figures.append(self._create_top_affected_graph(curr_data['top_affected'], dataset))
            # Create example figures, return first n_pictures_to_show from original and then n_pictures_to_show from
            # augmented dataset
            figures.append(self._create_example_figure(dataset, curr_data['images']))
            figures.append('<br>')

        return figures

    def _create_example_figure(self, dataset: VisionData, images):
        # Create tuple of ([base images], [aug images], [classes], <[bboxes]>)
        transposed = list(zip(*images))
        base_images = transposed[0]
        aug_images = transposed[1]
        classes = list(map(dataset.label_id_to_name, transposed[2]))

        # Create image figures
        fig = make_subplots(rows=2, cols=len(classes), column_titles=classes, row_titles=['Origin', 'Augmented'],
                            vertical_spacing=0.01, horizontal_spacing=0.01)

        # The width is accumulated and the height is taken by the max image
        images_width = 0
        max_height = 0
        for index, (base_image, aug_image) in enumerate(zip(base_images, aug_images)):
            # Add image figures
            fig.append_trace(numpy_to_image_figure(base_image), row=1, col=index + 1)
            fig.append_trace(numpy_to_image_figure(aug_image), row=2, col=index + 1)
            # Update sizes
            img_width, img_height = ImageInfo(base_image).get_size()
            images_width += img_width
            max_height = max(max_height, img_height)

        # Add 10 to space between image columns and another 20 for titles
        width = images_width + 10 * len(base_images) + 20
        # We have fixed 2 rows, and add 60 for titles space
        height = max_height * 2 + 60
        # Set minimum sizes in case of very small images
        height = max(height, 400)
        width = max(width, 800)
        # Font size is annoying and not fixed, but relative to image size, so set the base font relative to height
        base_font_size = height / 40

        # If length is 4 means we also have bounding boxes to draw
        if len(transposed) == 4:
            for index, (base_bbox, aug_bbox) in enumerate(transposed[3]):
                label_bbox_add_to_figure(base_bbox, fig, row=1, col=index + 1)
                label_bbox_add_to_figure(aug_bbox, fig, row=2, col=index + 1)

        (fig.update_layout(title=dict(text='Augmentation Samples', font=dict(size=base_font_size * 2)))
         .update_yaxes(showticklabels=False, visible=True, fixedrange=True, automargin=True)
         .update_xaxes(showticklabels=False, visible=True, fixedrange=True, automargin=True)
         .update_annotations(font_size=base_font_size * 1.5))

        # In case of heatmap (grayscale images), need to add those properties which on Image exists automatically
        if dataset.data_dimension == 1:
            apply_heatmap_image_properties(fig)

        return fig.to_image('svg', width=width, height=height).decode('utf-8')

    def _create_performance_graph(self, base_scores: dict, augmented_scores: dict):
        metrics = sorted(list(base_scores.keys()))
        fig = make_subplots(rows=1, cols=len(metrics), subplot_titles=metrics)

        for index, metric in enumerate(metrics):
            curr_aug = augmented_scores[metric]
            x = ['Origin', 'Augmented']
            y = [base_scores[metric], curr_aug['score']]
            diff = ['', format_percent(curr_aug['diff'])]

            fig.add_trace(go.Bar(x=x, y=y, customdata=diff, texttemplate='%{customdata}',
                                 textposition='auto'), col=index + 1, row=1)

        (fig.update_layout(font=dict(size=12), height=300, width=400 * len(metrics), autosize=False,
                           title=dict(text='Performance Comparison', font=dict(size=20)),
                           showlegend=False)
         .update_xaxes(title=None, type='category', tickangle=30))
        return fig

    def _create_top_affected_graph(self, top_affected_dict, dataset):
        metrics = sorted(top_affected_dict.keys())
        fig = make_subplots(rows=1, cols=len(metrics), subplot_titles=metrics)

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

            # Plotly have a bug that if all y values are zero text position 'auto' doesn't work
            textposition = 'outside' if sum(y) == 0 else 'auto'
            fig.add_trace(go.Bar(name=metric, x=x, y=y, customdata=custom_data, texttemplate='%{customdata[0]}',
                                 textposition=textposition, hovertemplate='Number of samples: %{customdata[1]}'),
                          row=1, col=index + 1)

        (fig.update_layout(font=dict(size=12), height=300, width=600 * len(metrics),
                           title=dict(text='Top Affected Classes', font=dict(size=20)),
                           showlegend=False)
         .update_xaxes(title=None, type='category', tickangle=30, tickprefix='Class ', automargin=True)
         .update_yaxes(automargin=True))

        return fig


def augmentation_name(aug):
    if isinstance(aug, imgaug.augmenters.Augmenter):
        return aug.name
    elif isinstance(aug, albumentations.BasicTransform):
        return aug.get_class_fullname()
    else:
        raise DeepchecksValueError(f'Unsupported augmentation type {type(aug)}')


def get_random_image_pairs_from_dataset(original_dataset: VisionData,
                                        augmented_dataset: VisionData,
                                        top_affected_classes: dict):
    """Get image pairs from 2 datasets.

    We iterate the internal dataset object directly to avoid randomness
    Dataset returns data points as processed images, making this currently not really usable
    To avoid making more assumptions this currently stays as-is
    Note that images return in RGB format, ond to visualize them using OpenCV the final dimension should be
    transposed;
    can be done via image = image[:, :, ::-1] or cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    """
    classes_to_show = {
        class_info['class']: class_info['diff']
        for classes_list in top_affected_classes.values()
        for class_info in classes_list
    }

    baseline_sampler = iter(original_dataset.get_data_loader().dataset)
    aug_sampler = iter(augmented_dataset.get_data_loader().dataset)
    samples = []
    # Will use the diff value to sort by highest diff first
    sort_value = []
    classes_set = set(classes_to_show.keys())
    # iterate and sample
    for (sample_base, sample_aug) in zip(baseline_sampler, aug_sampler):
        if not classes_set:
            break

        batch = original_dataset.to_batch(sample_base, sample_aug)
        batch_label: torch.Tensor = original_dataset.label_formatter(batch)
        images: List[np.ndarray] = original_dataset.image_formatter(batch)
        base_label: torch.Tensor = batch_label[0]
        aug_label: torch.Tensor = batch_label[1]
        if original_dataset.task_type == TaskType.OBJECT_DETECTION:
            # Classes are the first item in the label
            all_classes_in_label = set(
                base_label[:, 0].tolist() if len(base_label) > 0 else []
            )
            # If not relevant classes continue
            intersect = all_classes_in_label.intersection(classes_set)
            if not intersect:
                continue
            # Take randomly first class which will represents the current image
            curr_class = next(iter(intersect))
            # Take only bboxes of this class
            base_class_label = [x for x in base_label if x[0] == curr_class]
            aug_class_label = [x for x in aug_label if x[0] == curr_class]
            samples.append((images[0], images[1], curr_class, (base_class_label, aug_class_label)))
        elif original_dataset.task_type == TaskType.CLASSIFICATION:
            curr_class = base_label.item()
            if curr_class not in classes_set:
                continue
            samples.append((images[0], images[1], curr_class))
        else:
            raise DeepchecksValueError('Not implemented')

        # Add the sort value to sort later images by difference
        sort_value.append(classes_to_show[curr_class])
        # Remove from the classes set to not take another sample of the same class
        classes_set.remove(curr_class)

    # Sort by diff but return only the tuple
    return [s for s, _ in sorted(zip(samples, sort_value), key=lambda pair: pair[1])]
