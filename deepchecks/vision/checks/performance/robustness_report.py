"""Module containing robustness report check."""
from collections import defaultdict

import imgaug
from typing import TypeVar, List, Optional
import albumentations

import pandas as pd
import torch
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from ignite.metrics import Metric

from deepchecks import CheckResult, ConditionResult
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.vision import VisionData, SingleDatasetCheck, Context
from deepchecks.vision.dataset import TaskType
from deepchecks.vision.metrics_utils import calculate_metrics
from deepchecks.vision.utils.validation import set_seeds
from deepchecks.vision.metrics_utils import get_scorers_list
from deepchecks.utils.strings import format_percent
from deepchecks.vision.utils.base_formatters import BasePredictionFormatter
from deepchecks.vision.utils.image_functions import numpy_to_image_figure, apply_heatmap_image_properties, \
    is_images_equal

__all__ = ['RobustnessReport']


PR = TypeVar('PR', bound='RobustnessReport')


class RobustnessReport(SingleDatasetCheck):
    """Check several image enhancements for model robustness.

    Parameters
    ----------
        alternative_metrics (List[Metric], default None):
            A list of ignite.Metric objects whose score should be used. If None are given, use the default metrics.
    """

    def __init__(self,
                 prediction_formatter: BasePredictionFormatter,
                 alternative_metrics: Optional[List[Metric]] = None,
                 augmentations=None,
                 random_state: int = 42):
        super().__init__()
        self.alternative_metrics = alternative_metrics
        self.prediction_formatter = prediction_formatter
        self.random_state = random_state
        self.augmentations = augmentations

    def run_logic(self, context: Context, dataset_type: str = 'train') -> CheckResult:
        """Run check.

        Returns
        -------
            CheckResult: value is dictionary in format 'score-name': score-value
        """
        set_seeds(self.random_state)
        context.assert_task_type(TaskType.CLASSIFICATION, TaskType.OBJECT_DETECTION)
        if dataset_type == 'train':
            dataset = context.train
        else:
            dataset = context.test

        model = context.model

        # Validate the transformations works
        transforms_handler = dataset.get_transform_type()
        self._validate_augmenting_affects(transforms_handler, dataset)
        # Get default scorers if no alternative, or validate alternatives
        metrics = get_scorers_list(dataset, self.alternative_metrics)
        # Return dataframe of (Class, Metric, Value)
        base_results: pd.DataFrame = self._evaluate_dataset(dataset, metrics, model)
        # Return dict of metric to value
        base_mean_results: dict = self._calc_mean_metrics(base_results)
        # Get augmentations
        augmentations = self.augmentations or transforms_handler.get_robustness_augmentations(dataset.data_dimension)
        aug_all_data = {}
        for augmentation_func in augmentations:
            augmentation = augmentation_name(augmentation_func)
            aug_dataset = self._create_augmented_dataset(dataset, augmentation_func)
            # Return dataframe of (Class, Metric, Value)
            aug_results = self._evaluate_dataset(aug_dataset, metrics, model)
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

    def add_condition_degradation_not_more_than(self, ratio: 0.01):
        """Add condition which validates augmentations doesn't degrade the model metrics by given amount"""
        def condition(result):
            failed = [
                aug
                for aug, metrics in result.items()
                for metric, metric_data in metrics.items()
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
        """Validate the user is using the transforms' field correctly, and that if affects the image and label"""
        aug_dataset = self._create_augmented_dataset(dataset, transform_handler.get_test_transformation())
        # Iterate both datasets and compare results
        baseline_sampler = iter(dataset.get_data_loader().dataset)
        aug_sampler = iter(aug_dataset.get_data_loader().dataset)

        for (sample_base, sample_aug) in zip(baseline_sampler, aug_sampler):
            # Skips any sample without label
            if sample_base[1] is None or len(sample_base[1]) == 0:
                continue

            if is_images_equal(sample_base[0], sample_aug[0]):
                raise DeepchecksValueError('Found that images have not been affected by adding augmentation. '
                                           'This might be a problem with the implementation of Dataset.__getitem__')

            # For classification does not check label for difference
            if dataset.task_type != TaskType.CLASSIFICATION:
                label_a = dataset.label_transformer([sample_base[1]])[0]
                label_b = dataset.label_transformer([sample_aug[1]])[0]
                if torch.equal(label_a, label_b):
                    raise DeepchecksValueError('Found that labels have not been affected by adding augmentation. label '
                                               f'before {label_a} and after {label_b}. This might be a problem with the '
                                               f'implementation of Dataset.__getitem__')
            # If all validations passed return
            return

    def _evaluate_dataset(self, dataset: VisionData, metrics, model):
        classes = dataset.get_samples_per_class().keys()
        metrics_results = calculate_metrics(metrics, dataset, model, self.prediction_formatter)
        per_class_result = [
            [class_name, metric, class_score]
            for metric, score in metrics_results.items()
            # scorer returns numpy array of results with item per class
            for class_score, class_name in zip(score.tolist(), classes)
        ]

        return pd.DataFrame(per_class_result, columns=['Class', 'Metric', 'Value']).sort_values(by=['Class'])

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
                                                 'samples': dataset.get_samples_per_class()[index_class]})
        return aug_top_affected

    def _calc_performance_diff(self, mean_base, augmented_metrics):
        def difference(aug_score, base_score):
            # TODO how to handle zero base score
            return (aug_score - base_score) / base_score if base_score != 0 else 0

        diff_dict = {}
        for metric, score in self._calc_mean_metrics(augmented_metrics).items():
            diff_dict[metric] = {'score': score, 'diff': difference(score, mean_base[metric])}

        return diff_dict

    def _calc_mean_metrics(self, metrics_df) -> dict:
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
            figures.append(self._create_top_affected_graph(curr_data['top_affected']))
            # Create example figures, return first n_pictures_to_show from original and then n_pictures_to_show from
            # augmented dataset
            figures.append(self._create_example_figure(dataset, curr_data['images']))
            figures.append('<br>')

        return figures

    def _create_example_figure(self, dataset: VisionData, images):
        def to_batch(img_list):
            if isinstance(img_list[0], torch.Tensor):
                return torch.stack(img_list)
            return img_list
        # First join all images to convert them in a single action to displayable format
        # Create tuple of ([base images], [aug images], [classes])
        transposed = list(zip(*images))
        base_images = dataset.to_display_data(to_batch(transposed[0]))
        aug_images = dataset.to_display_data(to_batch(transposed[1]))
        classes = list(map(str, transposed[2]))
        dimension = dataset.data_dimension

        # Create image figures
        fig = make_subplots(rows=2, cols=len(classes), column_titles=classes, row_titles=['Origin', 'Augmented'],
                            vertical_spacing=0, horizontal_spacing=0)

        for index, (base_image, aug_image, curr_class) in enumerate(zip(base_images, aug_images, classes)):
            # Add image figures
            fig.append_trace(numpy_to_image_figure(base_image), row=1, col=index + 1)
            fig.append_trace(numpy_to_image_figure(aug_image), row=2, col=index + 1)

        (fig.update_layout(title=dict(text='Augmentation Samples', font=dict(size=20)),
                           margin=dict(l=0, r=0, t=60, b=0))
         .update_yaxes(showticklabels=False, visible=True, fixedrange=True)
         .update_xaxes(showticklabels=False, visible=True, fixedrange=True))

        # In case of heatmap and grayscale images, need to add those properties which on Image exists automatically
        if dimension == 1:
            apply_heatmap_image_properties(fig)

        return fig

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

    def _create_top_affected_graph(self, top_affected_dict):
        metrics = sorted(top_affected_dict.keys())
        fig = make_subplots(rows=1, cols=len(metrics), subplot_titles=metrics)

        for index, metric in enumerate(metrics):
            metric_classes = top_affected_dict[metric]
            # Take the worst affected classes
            x = []
            y = []
            custom_data = []
            for class_info in metric_classes:
                x.append(class_info['class'])
                y.append(class_info['value'])
                custom_data.append([format_percent(class_info['diff']), class_info['samples']])

            fig.add_trace(go.Bar(name=metric, x=x, y=y, customdata=custom_data, texttemplate='%{customdata[0]}',
                                 textposition='auto', hovertemplate='Number of samples: %{customdata[1]}'),
                          row=1, col=index + 1)

        (fig.update_layout(font=dict(size=12), height=300, width=600 * len(metrics),
                           title=dict(text='Top Affected Classes', font=dict(size=20)),
                           showlegend=False)
         .update_xaxes(title=None, type='category', tickangle=30, tickprefix='Class '))

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

    # We definitely make the assumption that the underlying structure is torch.utils.Dataset
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
        curr_classes: set = original_dataset.label_transformer.get_classes(sample_base[1])
        # If not relevant classes continue
        intersect = curr_classes.intersection(classes_set)
        if not intersect:
            continue
        # Take randomly first class which will represents the current image
        first_class = next(iter(intersect))
        samples.append((sample_base[0], sample_aug[0], first_class))
        sort_value.append(classes_to_show[first_class])
        classes_set.remove(first_class)

    # Sort by diff but return only the tuple
    return [s for s, _ in sorted(zip(samples, sort_value), key=lambda pair: pair[1])]
