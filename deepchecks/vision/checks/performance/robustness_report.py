"""Module containing robustness report check."""
from collections import defaultdict

import imgaug
from typing import Callable, TypeVar, List, Optional
import albumentations

import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from ignite.metrics import Metric


from deepchecks import CheckResult
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.vision import VisionDataset, SingleDatasetCheck, Context
from deepchecks.vision.dataset import TaskType
from deepchecks.vision.metrics_utils import calculate_metrics
from deepchecks.vision.utils.validation import set_seeds
from deepchecks.vision.utils.transformations import TransformWrapper
from deepchecks.vision.metrics_utils import get_scorers_list
from deepchecks.utils.strings import format_percent

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
                 alternative_metrics: Optional[List[Metric]] = None,
                 prediction_extract: Optional[Callable] = None,
                 augmentations=None,
                 epsilon: float = 10 ** -2,
                 random_state: int = 42):
        super().__init__()
        self._epsilon = epsilon
        self.alternative_metrics = alternative_metrics
        self.prediction_extract = prediction_extract
        self.random_state = random_state
        self.augmentations = augmentations

    def run_logic(self, context: Context, dataset_type: str = 'train') -> CheckResult:
        """Run check.

        Returns
        -------
            CheckResult: value is dictionary in format 'score-name': score-value
        """
        set_seeds(self.random_state)
        context.assert_task_type(TaskType.CLASSIFICATION)
        if dataset_type == 'train':
            dataset = context.train
        else:
            dataset = context.test

        model = context.model

        # Get default scorers if no alternative, or validate alternatives
        metrics = get_scorers_list(dataset, self.alternative_metrics)

        base_results: pd.DataFrame = self._evaluate_dataset(dataset, metrics, model)
        aug_results, examples = self._augment_and_evaluate(dataset, model, metrics)

        # Evaluate dataset return score per class, Then creating dict of mean scores for each metric + augmentation
        base_mean_scorers, scores_diff_dict = self._create_performance_dicts(base_results, aug_results)
        # Create figures to display
        performance = self._create_performance_graph(base_mean_scorers, scores_diff_dict)
        histogram = self._create_class_histogram(base_results, aug_results, dataset)
        examples_figures = self._create_example_figure(dataset, examples)

        return CheckResult(
            scores_diff_dict,
            header='Robustness Report',
            display=['<h3>Performance Comparison</h3>', performance, '<h3>Top Affected Classes</h3>',
                     histogram, '<h3>Examples</h3>', *examples_figures]
        )

    def _augment_and_evaluate(self, dataset: VisionDataset, model, metrics):
        augmentations = self.augmentations or get_robustness_augmentations(dataset.get_transform_type())
        examples = {}
        results = {}

        for curr_aug in augmentations:
            # Create a copy of data loader and the dataset
            aug_dataset: VisionDataset = dataset.copy()
            transform: TransformWrapper = aug_dataset.wrap_transform_field()
            # Add augmentation in the first place
            transform.add_augmentation_in_start(curr_aug)
            # Evaluate
            results_aug_df = self._evaluate_dataset(aug_dataset, metrics, model)
            # Add augmentation name
            aug_name = augmentation_name(curr_aug)
            results[aug_name] = results_aug_df
            examples[aug_name] = get_random_image_pairs_from_dataset(dataset, aug_dataset)

        return results, examples

    def _evaluate_dataset(self, dataset: VisionDataset, metrics, model):
        classes = dataset.get_samples_per_class().keys()
        metrics_results = calculate_metrics(metrics, dataset, model, self.prediction_extract)
        per_class_result = (
            [class_name, metric, class_score]
            for metric, score in metrics_results.items()
            # scorer returns numpy array of results with item per class
            for class_score, class_name in zip(score.tolist(), classes)
        )

        return pd.DataFrame(per_class_result, columns=['Class', 'Metric', 'Value']).sort_values(by=['Class'])

    def _create_example_figure(self, dataset, example_dict):
        figures = []

        for aug_name, images in example_dict.items():
            fig = make_subplots(rows=2, cols=len(images), horizontal_spacing=0.1, vertical_spacing=0.1,
                                subplot_titles=['Original'] * len(images) + ['Augmented'] * len(images))
            height = 0
            width = 0
            for index, (base_image, aug_image) in enumerate(images):
                height = len(base_image[0])
                width = len(base_image[0][0])
                fig.add_trace(go.Image(z=dataset.display_transform(base_image), hoverinfo='skip'), row=1, col=index + 1)
                fig.add_trace(go.Image(z=dataset.display_transform(aug_image), hoverinfo='skip'), row=2, col=index + 1)

            fig.update_layout(title_text=aug_name, height=height * 2, width=width * len(images))
            fig.update_yaxes(showticklabels=False, visible=True, fixedrange=True)
            fig.update_xaxes(showticklabels=False, visible=True, fixedrange=True)
            figures.append(fig)

        return figures

    def _create_class_histogram(self, base, augmented, dataset: VisionDataset, n_classes_to_show=5):
        def calc_percent(a, b):
            return (a - b) / b if b != 0 else 0

        graph_data = []
        for aug_name, class_scores in augmented.items():
            for metric in class_scores['Metric'].unique():
                single_metric_scores = class_scores[class_scores['Metric'] == metric][['Class', 'Value']]\
                    .set_index('Class')
                single_metric_scores['Base'] = base[base['Metric'] == metric][['Class', 'Value']].set_index('Class')
                diff = single_metric_scores.apply(lambda x: calc_percent(x.Value, x.Base), axis=1)
                # Take the worst affected classes
                for index_class, diff_value in diff.sort_values()[:n_classes_to_show].iteritems():
                    graph_data.append([index_class, aug_name, metric,
                                       format_percent(diff_value),
                                       single_metric_scores.at[index_class, 'Value'],
                                       dataset.get_samples_per_class()[index_class]])

        fig = px.bar(
            pd.DataFrame(graph_data, columns=['Class', 'Augmentation', 'Metric', 'Difference', 'Value',
                                              'Sample Count']),
            x='Class',
            y='Value',
            color='Augmentation',
            barmode='group',
            facet_col='Augmentation',
            facet_row='Metric',
            facet_col_spacing=0.05,
            facet_row_spacing=0.1,
            text='Difference',
            hover_data=['Sample Count']
        )

        (fig.update_xaxes(tickprefix='Class ')
            .update_xaxes(title=None, type='category', matches=None)
            .update_yaxes(title=None, matches=None)
            .for_each_annotation(lambda a: a.update(text=a.text.split('=')[-1]))
            .for_each_yaxis(lambda yaxis: yaxis.update(showticklabels=True))
            .for_each_xaxis(lambda xaxis: xaxis.update(showticklabels=True))
            .update_traces(textposition='outside'))

        return fig

    def _create_performance_dicts(self, base, augmented):
        def get_metrics(df) -> dict:
            df = df[['Metric', 'Value']].groupby(['Metric']).median()
            return df.to_dict()['Value']

        def difference(aug_score, base_score):
            return (aug_score - base_score) / base_score

        # Get dict of metric to score
        original_scores = get_metrics(base)

        diff_dict = defaultdict(dict)
        for aug_name, df in augmented.items():
            for metric, score in get_metrics(df).items():
                diff_dict[aug_name][metric] = {'score': score, 'diff': difference(score, original_scores[metric])}

        return original_scores, diff_dict

    def _create_performance_graph(self, original_scores: dict, augmented_scores: dict):
        # Get dict of metric to score
        graph_data = [
            [metric, scores_info['score'], aug_name, format_percent(scores_info['diff'])]
            for aug_name, metrics in augmented_scores.items()
            for metric, scores_info in metrics.items()
        ]
        graph_data.extend([[metric, score, 'Origin', None] for metric, score in original_scores.items()])

        fig = px.bar(
            pd.DataFrame(graph_data, columns=['Metric', 'Value', 'Augmentation', 'Difference']),
            x='Augmentation',
            y='Value',
            color='Augmentation',
            barmode='group',
            facet_col='Metric',
            facet_col_spacing=0.05,
            text='Difference',
            width=400
        )

        (fig.update_layout()
            .update_xaxes(title=None, type='category', tickangle=60)
            .update_yaxes(title=None, matches=None)
            .for_each_annotation(lambda a: a.update(text=a.text.split('=')[-1]))
            .for_each_yaxis(lambda yaxis: yaxis.update(showticklabels=True))
            .update_traces(textposition='inside'))

        return fig


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
    elif transform_type == 'imgaug':
        return [
            imgaug.augmenters.MultiplyHueAndSaturation()
        ]
    else:
        raise DeepchecksValueError(f'Transformations of type {transform_type} are not supported')


def augmentation_name(aug):
    if isinstance(aug, imgaug.augmenters.Augmenter):
        return aug.name
    elif isinstance(aug, albumentations.BasicTransform):
        return aug.get_class_fullname()
    else:
        raise DeepchecksValueError(f'Unsupported augmentation type {type(aug)}')


def get_random_image_pairs_from_dataset(original_dataset: VisionDataset,
                                        augmented_dataset: VisionDataset,
                                        n_samples=5):
    """Get image pairs from 2 datasets.

    We iterate the internal dataset object directly to avoid randomness
    Dataset returns data points as processed images, making this currently not really usable
    To avoid making more assumptions this currently stays as-is
    Note that images return in RGB format, ond to visualize them using OpenCV the final dimension should be
    transposed;
    can be done via image = image[:, :, ::-1] or cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    """
    # We definitely make the assumption that the underlying structure is torch.utils.Dataset
    baseline_sampler = iter(original_dataset.get_data_loader().dataset)
    aug_sampler = iter(augmented_dataset.get_data_loader().dataset)
    samples = []
    # iterate and sample
    for idx, (sample_base, sample_aug) in enumerate(zip(baseline_sampler, aug_sampler)):
        if idx >= n_samples:
            break
        samples.append((sample_base[0], sample_aug[0]))

    return samples
