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
"""Module containing class performance check."""
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple, TypeVar

import numpy as np
import pandas as pd
import plotly.express as px
import torch
from ignite.metrics import Metric
from sklearn.preprocessing import LabelBinarizer

from deepchecks.core import CheckResult, ConditionResult, DatasetKind
from deepchecks.core.condition import ConditionCategory
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.utils import plot
from deepchecks.utils.strings import format_number, format_percent
from deepchecks.vision import Batch, Context, TrainTestCheck
from deepchecks.vision.metrics_utils.iou_utils import jaccard_iou
from deepchecks.vision.metrics_utils.metrics import (
    filter_classes_for_display, get_scorers_list, metric_results_to_df)
from deepchecks.vision.task_type import TaskType
from deepchecks.vision.utils.image_functions import (prepare_grid,
                                                     prepare_sample_thumbnail,
                                                     prepare_thumbnail)

__all__ = ['ClassPerformance']


PR = TypeVar('PR', bound='ClassPerformance')


class ClassPerformance(TrainTestCheck):
    """Summarize given metrics on a dataset and model.

    Parameters
    ----------
    alternative_metrics : Dict[str, Metric], default: None
        A dictionary of metrics, where the key is the metric name and the value is an ignite.Metric object whose score
        should be used. If None are given, use the default metrics.
    n_to_show : int, default: 20
        Number of classes to show in the report. If None, show all classes.
    show_only : str, default: 'largest'
        Specify which classes to show in the report. Can be one of the following:
        - 'largest': Show the largest classes.
        - 'smallest': Show the smallest classes.
        - 'random': Show random classes.
        - 'best': Show the classes with the highest score.
        - 'worst': Show the classes with the lowest score.
    metric_to_show_by : str, default: None
        Specify the metric to sort the results by. Relevant only when show_only is 'best' or 'worst'.
        If None, sorting by the first metric in the default metrics list.
    class_list_to_show: List[int], default: None
        Specify the list of classes to show in the report. If specified, n_to_show, show_only and metric_to_show_by
        are ignored.
    """

    def __init__(
        self,
        alternative_metrics: Optional[Dict[str, Metric]] = None,
        n_to_show: int = 20,
        show_only: str = 'largest',
        metric_to_show_by: Optional[str] = None,
        class_list_to_show: Optional[List[int]] = None,
        n_of_images: int = 5,
        thumbnail_size: Tuple[int, int] = (400, 400),
        bbox_border_width: int = 2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.alternative_metrics = alternative_metrics
        self.n_to_show = n_to_show
        self.class_list_to_show = class_list_to_show
        self.n_of_images = n_of_images
        self.thumbnail_size = thumbnail_size
        self.bbox_border_width = bbox_border_width

        if self.class_list_to_show is None:
            if show_only not in ['largest', 'smallest', 'random', 'best', 'worst']:
                raise DeepchecksValueError(f'Invalid value for show_only: {show_only}. Should be one of: '
                                           f'["largest", "smallest", "random", "best", "worst"]')

            self.show_only = show_only
            if alternative_metrics is not None and show_only in ['best', 'worst'] and metric_to_show_by is None:
                raise DeepchecksValueError('When alternative_metrics are provided and show_only is one of: '
                                           '["best", "worst"], metric_to_show_by must be specified.')

        self.metric_to_show_by = metric_to_show_by
        self._data_metrics = {}
        # are used in case of classification task type
        # var type - list[tuple[image, y-true, y-predicted]]]
        self._successfully_evaluated_images: Optional[List[Tuple[np.ndarray, int, int]]] = None
        self._unsuccessfully_evaluated_images: Optional[List[Tuple[np.ndarray, int, int]]] = None
        # is used in case of detection task type
        self._images: Optional[List[BboxesMatchResult]] = None

    def initialize_run(self, context: Context):
        """Initialize run by creating the _state member with metrics for train and test."""
        self._data_metrics = {}
        self._data_metrics[DatasetKind.TRAIN] = get_scorers_list(context.train, self.alternative_metrics)
        self._data_metrics[DatasetKind.TEST] = get_scorers_list(context.train, self.alternative_metrics)

        if context.train.task_type is TaskType.CLASSIFICATION:
            self._successfully_evaluated_images = []
            self._unsuccessfully_evaluated_images = []
        elif context.train.task_type is TaskType.OBJECT_DETECTION:
            self._images = []

        if not self.metric_to_show_by:
            self.metric_to_show_by = list(self._data_metrics[DatasetKind.TRAIN].keys())[0]

    def update(self, context: Context, batch: Batch, dataset_kind: DatasetKind):
        """Update the metrics by passing the batch to ignite metric update method."""
        label = batch.labels
        prediction = batch.predictions

        for _, metric in self._data_metrics[dataset_kind].items():
            metric.update((prediction, label))

        self._collect_output_images(
            context=context,
            batch=batch,
            dataset_kind=dataset_kind
        )

    def compute(self, context: Context) -> CheckResult:
        """Compute the metric result using the ignite metrics compute method and create display."""
        results = []

        for dataset_kind in [DatasetKind.TRAIN, DatasetKind.TEST]:
            dataset = context.get_data_by_kind(dataset_kind)
            computed_metrics = {k: m.compute() for k, m in self._data_metrics[dataset_kind].items()}
            metrics_df = metric_results_to_df(computed_metrics, dataset)
            metrics_df['Dataset'] = dataset_kind.value
            metrics_df['Number of samples'] = metrics_df['Class'].map(dataset.n_of_samples_per_class.get)
            results.append(metrics_df)

        results_df = pd.concat(results)[[
            'Dataset',
            'Metric',
            'Class',
            'Class Name',
            'Number of samples',
            'Value'
        ]]

        if self.class_list_to_show is not None:
            results_df = results_df.loc[results_df['Class'].isin(self.class_list_to_show)]

        elif self.n_to_show is not None:
            classes_to_show = filter_classes_for_display(
                results_df,
                self.metric_to_show_by,
                self.n_to_show,
                self.show_only
            )
            results_df = results_df.loc[results_df['Class'].isin(classes_to_show)]

        results_df = results_df.sort_values(by=['Dataset', 'Value'], ascending=False)
        fig = px.histogram(
            results_df,
            x='Class Name',
            y='Value',
            color='Dataset',
            color_discrete_sequence=(plot.colors['Train'], plot.colors['Test']),
            barmode='group',
            facet_col='Metric',
            facet_col_spacing=0.05,
            hover_data=['Number of samples'],
        )

        fig = (
            fig
            .update_xaxes(title='Class', type='category')
            .update_yaxes(title='Value', matches=None)
            .for_each_annotation(lambda a: a.update(text=a.text.split('=')[-1]))
            .for_each_yaxis(lambda yaxis: yaxis.update(showticklabels=True))
        )

        task_type = context.train.task_type

        if task_type is TaskType.CLASSIFICATION:
            display = [fig, self._prepare_images_thumbnails_for_classification_task()]
        elif task_type is TaskType.OBJECT_DETECTION:
            display = [fig, self._prepare_images_thumbnails_for_detection_task(context)]
        else:
            display = [fig]

        return CheckResult(
            results_df,
            header='Class Performance',
            display=display
        )

    def add_condition_test_performance_not_less_than(self: PR, min_score: float) -> PR:
        """Add condition - metric scores are not less than given score.

        Parameters
        ----------
        min_score : float
            Minimum score to pass the check.
        """
        def condition(check_result: pd.DataFrame):
            not_passed = check_result.loc[check_result['Value'] < min_score]
            not_passed_test = check_result.loc[check_result['Dataset'] == 'Test']
            if len(not_passed):
                details = f'Found metrics with scores below threshold:\n' \
                          f'{not_passed_test[["Class Name", "Metric", "Value"]].to_dict("records")}'
                return ConditionResult(ConditionCategory.FAIL, details)
            return ConditionResult(ConditionCategory.PASS)

        return self.add_condition(f'Scores are not less than {min_score}', condition)

    def add_condition_train_test_relative_degradation_not_greater_than(self: PR, threshold: float = 0.1) -> PR:
        """Add condition that will check that test performance is not degraded by more than given percentage in train.

        Parameters
        ----------
        threshold : float
            maximum degradation ratio allowed (value between 0 and 1)
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

            if check_result.get('Class Name') is not None:
                classes = check_result['Class Name'].unique()
            else:
                classes = None
            explained_failures = []
            if classes is not None:
                for class_name in classes:
                    test_scores_class = test_scores.loc[test_scores['Class Name'] == class_name]
                    train_scores_class = train_scores.loc[train_scores['Class Name'] == class_name]
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
                return ConditionResult(ConditionCategory.FAIL, message)
            else:
                return ConditionResult(ConditionCategory.PASS)

        return self.add_condition(f'Train-Test scores relative degradation is not greater than {threshold}',
                                  condition)

    def add_condition_class_performance_imbalance_ratio_not_greater_than(
        self: PR,
        threshold: float  = 0.3,
        score: Optional[str] = None
    ) -> PR:
        """Add condition.

        Verifying that relative ratio difference
        between highest-class and lowest-class is not greater than 'threshold'.

        Parameters
        ----------
        threshold : float
            ratio difference threshold
        score : str
            limit score for condition

        Returns
        -------
        Self
            instance of 'ClassPerformance' or it subtype

        Raises
        ------
        DeepchecksValueError
            if unknown score function name were passed;
        """
        if score is None:
            raise DeepchecksValueError('Must define "score" parameter')

        def condition(check_result: pd.DataFrame) -> ConditionResult:
            if score not in set(check_result['Metric']):
                raise DeepchecksValueError(f'Data was not calculated using the scoring function: {score}')

            datasets_details = []
            for dataset in ['Test', 'Train']:
                data = check_result.loc[(check_result['Dataset'] == dataset) & (check_result['Metric'] == score)]

                min_value_index = data['Value'].idxmin()
                min_row = data.loc[min_value_index]
                min_class_name = min_row['Class Name']
                min_value = min_row['Value']

                max_value_index = data['Value'].idxmax()
                max_row = data.loc[max_value_index]
                max_class_name = max_row['Class Name']
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
                return ConditionResult(ConditionCategory.FAIL, details='\n'.join(datasets_details))
            else:
                return ConditionResult(ConditionCategory.PASS)

        return self.add_condition(
            name=(
                f'Relative ratio difference between labels \'{score}\' score '
                f'is not greater than {format_percent(threshold)}'
            ),
            condition_func=condition
        )

    def _collect_output_images(
        self,
        context: Context,
        batch: Batch,
        dataset_kind: DatasetKind
    ):
        dataset = context.get_data_by_kind(dataset_kind)
        task_type = dataset.task_type

        if task_type is TaskType.CLASSIFICATION:
            self._collect_output_images_for_classification_task(batch)
        elif task_type is TaskType.OBJECT_DETECTION:
            self._collect_output_images_for_detection_task(batch)
        else:
            # Do nothing in this case
            pass

    def _collect_output_images_for_classification_task(
        self,
        batch: Batch,
    ):
        assert (
            self._successfully_evaluated_images is not None
            and self._unsuccessfully_evaluated_images is not None
        )

        detected_classes = (
            LabelBinarizer()
            .fit(batch.labels)
            .inverse_transform(batch.predictions)
        )

        for img, y_true, y_pred in zip(batch.images, batch.labels, detected_classes):
            sample = (
                img,
                int(y_true.item()),
                int(y_pred)  # type: ignore
            )
            if y_true == y_pred and len(self._successfully_evaluated_images) < self.n_of_images:
                self._successfully_evaluated_images.append(sample)
            elif y_true != y_pred and len(self._unsuccessfully_evaluated_images) < self.n_of_images:
                self._unsuccessfully_evaluated_images.append(sample)

    def _prepare_images_thumbnails_for_classification_task(self) -> str:
        assert (
            self._successfully_evaluated_images is not None
            and self._unsuccessfully_evaluated_images is not None
        )

        description_grid_style = {
            'align-self': 'start',
            'padding': '0',
            'padding-top': '2rem',
        }

        if len(self._successfully_evaluated_images) == 0:
            successful_evaluation = '<p>Nothing to show</p>'
        else:
            successful_evaluation_thumbnails = []
            successful_evaluation_description = []

            for img, y_true, y_pred in self._successfully_evaluated_images:
                successful_evaluation_thumbnails.append(prepare_thumbnail(
                    image=img, size=self.thumbnail_size
                ))
                successful_evaluation_description.append(prepare_grid(
                    n_of_rows=1,
                    n_of_columns=2,
                    style=description_grid_style,
                    content=['<span><b>y-true</b></span>', f'<span>{y_true}</span>']
                ))

            successful_evaluation = prepare_grid(
                style={
                    'grid-template-rows': 'auto auto',
                    'grid-template-columns': f'repeat({len(successful_evaluation_thumbnails)}, 1fr)',
                },
                content=[*successful_evaluation_thumbnails, *successful_evaluation_description]
            )

        if len(self._unsuccessfully_evaluated_images) == 0:
            unsuccessful_evaluation = '<p>Nothing to show</p>'
        else:
            unsuccessful_evaluation_thumbnails = []
            unsuccessful_evaluation_description = []

            for img, y_true, y_pred in self._unsuccessfully_evaluated_images:
                unsuccessful_evaluation_thumbnails.append(prepare_thumbnail(
                    image=img,
                    size=self.thumbnail_size
                ))
                unsuccessful_evaluation_description.append(prepare_grid(
                    n_of_rows=2,
                    n_of_columns=2,
                    style=description_grid_style,
                    content=[
                        '<span><b>y-true</b></span>',
                        '<span><b>y-pred</b></span>',
                        f'<span>{y_true}</span>',
                        f'<span>{y_pred}</span>'
                    ]
                ))

            unsuccessful_evaluation = prepare_grid(
                style={
                    'grid-template-rows': 'auto auto',
                    'grid-template-columns': f'repeat({len(unsuccessful_evaluation_thumbnails)}, 1fr)',
                },
                content=[*unsuccessful_evaluation_thumbnails, *unsuccessful_evaluation_description]
            )

        content_template = (
            '<h5><b>Correct model predictions</b></h5>'
            '<div style="padding-left: 1rem;">{successful_evaluation}</div>'
            '<h5><b>Incorrect model predictions</b></h5>'
            '<div style="padding-left: 1rem;">{unsuccessful_evaluation}</div>'
        )
        return content_template.format(
            successful_evaluation=successful_evaluation,
            unsuccessful_evaluation=unsuccessful_evaluation
        )

    def _collect_output_images_for_detection_task(
        self,
        batch: Batch,
    ):
        assert self._images is not None

        def are_enough_images(stat: List[BboxesMatchResult]) -> bool:
            return len([
                it for it in stat
                if (
                    len(it.matches) != 0
                    and len(it.no_overlapping_dt) != 0
                    and len(it.no_overlapping_gt) != 0
                )
            ]) >= self.n_of_images

        if are_enough_images(self._images):
            return

        for img, gt, dt in zip(batch.images, batch.labels, batch.predictions):
            if not are_enough_images(self._images):
                self._images.append(match_bboxes(img, gt, dt))
            else:
                return

    def _prepare_images_thumbnails_for_detection_task(
        self,
        context: Context
    ):
        assert self._images is not None

        content = []
        class_name = context.train.label_id_to_name

        section_template = (
            '<h5><b>{title}</b></h5>'
            '<div style="padding-left: 1rem;">{content}</div>'
        )

        for index, stat in enumerate(self._images[:self.n_of_images], start=1):
            content.append(''.join((
                f'<h5><b>Image #{index}</b></h5>',
                # ==
                section_template.format(
                    title='<h5><b>Correctly detected objects</b></h5>',
                    content=(
                        self._prepare_grid_of_detected_objects(
                            stat.image,
                            stat.matches,
                            class_name)
                        if len(stat.matches) != 0
                        else '<p>nothing to show</p>')),
                # ==
                section_template.format(
                    title='<h5><b>No overlapping ground truth</b></h5>',
                    content=(
                        self._prepare_grid_of_no_overlapping_objects(
                            img=stat.image,
                            bboxes=stat.no_overlapping_gt,
                            class_name=class_name)
                        if len(stat.no_overlapping_gt) != 0
                        else '<p>nothing to show</p>')),
                # ==
                section_template.format(
                    title='<h5><b>No overlapping detected truth</b></h5>',
                    content=(
                        self._prepare_grid_of_no_overlapping_objects(
                            img=stat.image,
                            bboxes=stat.no_overlapping_dt,
                            class_name=class_name)
                        if len(stat.no_overlapping_dt) != 0
                        else '<p>nothing to show</p>'))
            )))

        return '<hr style="margin-bottom: 1rem; border: 1px dashed #999;">'.join(content)

    def _prepare_grid_of_detected_objects(
        self,
        img: np.ndarray,
        bbox_pairs: List['BBoxPair'],
        class_name: Callable[[int], str],
    ) -> str:
        images_tags = []
        info = []

        for gt, dt, ious in bbox_pairs:
            dt_class_id = dt[-1]
            gt_class_id = gt[0]
            images_tags.append(prepare_sample_thumbnail(
                image=img,
                gt=gt,
                dt=dt,
                border_width=self.bbox_border_width,
                size=self.thumbnail_size,
                include_label=False
            ))
            info.append(prepare_grid(
                n_of_columns=3,
                n_of_rows=2,
                style={
                    'align-self': 'start',
                    'padding': '0',
                    'padding-top': '2rem'
                },
                content=[
                    '<span><b>ground truth (red)</b></span>',
                    '<span><b>detected truth (blue)</b></span>',
                    '<span><b>ious</b></span>',
                    f'<p>{class_name(gt_class_id)}</p>',
                    f'<p>{class_name(dt_class_id)}</p>',
                    f'<p>{format_number(ious, 5)}</p>'
                ],
            ))

        return prepare_grid(
            style={
                'grid-template-rows': 'auto auto',
                'grid-template-columns': f'repeat({len(images_tags)}, 1fr)',
            },
            content=[*images_tags, *info]
        )

    def _prepare_grid_of_no_overlapping_objects(
        self,
        img: np.ndarray,
        bboxes: np.ndarray,
        class_name: Callable[[int], str],
        is_ground_truth: bool = True,
    ) -> str:
        images_tags = []
        info = []

        thumbnail_kwargs = {
            'image': img,
            'border_width': self.bbox_border_width,
            'size': self.thumbnail_size,
            'include_label': False
        }

        for bbox in bboxes:

            if is_ground_truth:
                title_tag = '<span><b>ground truth</b></span>'
                class_name_tag = f'<span>{class_name(bbox[0])}</span>'
                thumbnail_kwargs['gt'] = bbox
            else:
                title_tag = '<span><b>detected truth</b></span>'
                class_name_tag = f'<span>{class_name(bbox[-1])}</span>'
                thumbnail_kwargs['dt'] = bbox

            images_tags.append(prepare_sample_thumbnail(
                **thumbnail_kwargs
            ))
            info.append(prepare_grid(
                n_of_columns=2,
                n_of_rows=1,
                style={'align-self': 'start'},
                content=[title_tag, class_name_tag,],
            ))

        return prepare_grid(
            style={
                'grid-template-rows': 'auto auto',
                'grid-template-columns': f'repeat({len(images_tags)}, 1fr)',
            },
            content=[*images_tags, *info]
        )

# TODO: functionality below could be resuded in confusion matrix


class BBoxPair(NamedTuple):
    gt: np.ndarray
    dt: np.ndarray
    ious: float


class BboxesMatchResult(NamedTuple):
    image: np.ndarray
    matches: List[BBoxPair]
    no_overlapping_gt: np.ndarray  # array[gt-bbox, ...]
    no_overlapping_dt: np.ndarray  # array[dt-bbox, ...]


def match_bboxes(
    image: np.ndarray,
    ground_truth: torch.Tensor,
    detected_truth: torch.Tensor,
    ious_threshold: float = 0.5,
    confidence_threshold: float = 0
) -> BboxesMatchResult:
    passed_threshold_dt = np.array([
        detection.cpu().detach().numpy()
        for detection in detected_truth
        if detection[4] > confidence_threshold
    ])

    if len(passed_threshold_dt) == 0:
        return BboxesMatchResult(
            image=image,
            matches=[],
            no_overlapping_gt=ground_truth.cpu().detach().numpy(),
            no_overlapping_dt=np.array([]),
        )

    iter_of_ious = (
        (
            gt_idx,
            dt_idx,
            jaccard_iou(
                dt_bbox,  # is a numpy array already
                gt_bbox.cpu().detach().numpy()
            )
        )
        for gt_idx, gt_bbox in enumerate(ground_truth)
        for dt_idx, dt_bbox in enumerate(passed_threshold_dt)
    )
    matches = np.array([
        [gt_idx, dt_idx, ious]
        for gt_idx, dt_idx, ious in iter_of_ious
        if ious > ious_threshold
    ])

    # remove duplicate matches
    if len(matches) > 0:
        # sort by ious, in descend order
        matches = matches[matches[:, 2].argsort()[::-1]]
        # leave matches with unique prediction and the highest ious
        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
        # sort by ious, in descend order
        matches = matches[matches[:, 2].argsort()[::-1]]
        # leave matches with unique label and the highest ious
        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]

    if len(matches) == 0:
        return BboxesMatchResult(
            image=image,
            matches=[],
            no_overlapping_gt=ground_truth.cpu().detach().numpy(),
            no_overlapping_dt=passed_threshold_dt  # is a numpy array already
        )

    prepared_matches = [
        BBoxPair(
            gt=ground_truth[int(gt_idx)].cpu().detach().numpy(),
            dt=passed_threshold_dt[int(dt_idx)],  # is a numpy array already
            ious=ious
        )
        for gt_idx, dt_idx, ious in matches
    ]
    no_overlapping_gt = [
        ground_truth[gt_idx]
        for gt_idx in range(len(ground_truth))
        if (matches[:, 0] == gt_idx).any() is False
    ]
    no_overlapping_dt = [
        passed_threshold_dt[dt_idx]
        for dt_idx in range(len(passed_threshold_dt))
        if (matches[:, 1] == dt_idx).any() is False
    ]

    return BboxesMatchResult(
        image=image,
        matches=prepared_matches,
        no_overlapping_gt=(
            torch.stack(no_overlapping_gt).cpu().detach().numpy()
            if no_overlapping_gt
            else np.array([])
        ),
        no_overlapping_dt=(
            torch.stack(no_overlapping_dt).cpu().detach().numpy()
            if no_overlapping_gt
            else np.array([])
        )
    )
