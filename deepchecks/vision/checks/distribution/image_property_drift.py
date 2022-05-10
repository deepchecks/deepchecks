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
"""Module contains Image Property Drift check."""
import typing as t
import warnings
from collections import defaultdict
from textwrap import dedent

import pandas as pd
import PIL.Image as pilimage

from deepchecks.core import CheckResult, ConditionResult, DatasetKind
from deepchecks.core.condition import ConditionCategory
from deepchecks.core.errors import DeepchecksValueError, NotEnoughSamplesError
from deepchecks.utils.distribution.drift import calc_drift_and_plot
from deepchecks.utils.strings import format_number
from deepchecks.vision import Batch, Context, TrainTestCheck
from deepchecks.vision.utils.image_functions import (prepare_grid,
                                                     prepare_thumbnail)
from deepchecks.vision.utils.image_properties import (default_image_properties,
                                                      get_column_type,
                                                      validate_properties)

__all__ = ['ImagePropertyDrift']


TImagePropertyDrift = t.TypeVar('TImagePropertyDrift', bound='ImagePropertyDrift')


class ImagePropertyDrift(TrainTestCheck):
    """
    Calculate drift between train dataset and test dataset per image property, using statistical measures.

    Check calculates a drift score for each image property in test dataset, by comparing its distribution to the train
    dataset. For this, we use the Earth Movers Distance.

    See https://en.wikipedia.org/wiki/Wasserstein_metric

    Parameters
    ----------
    image_properties : List[Dict[str, Any]], default: None
        List of properties. Replaces the default deepchecks properties.
        Each property is dictionary with keys 'name' (str), 'method' (Callable) and 'output_type' (str),
        representing attributes of said method. 'output_type' must be one of 'continuous'/'discrete'
    margin_quantile_filter: float, default: 0.025
        float in range [0,0.5), representing which margins (high and low quantiles) of the distribution will be filtered
        out of the EMD calculation. This is done in order for extreme values not to affect the calculation
        disproportionally. This filter is applied to both distributions, in both margins.
    max_num_categories_for_drift: int, default: 10
        Only for non-continuous properties. Max number of allowed categories. If there are more,
        they are binned into an "Other" category. If None, there is no limit.
    max_num_categories_for_display: int, default: 10
        Max number of categories to show in plot.
    show_categories_by: str, default: 'largest_difference'
        Specify which categories to show for categorical features' graphs, as the number of shown categories is limited
        by max_num_categories_for_display. Possible values:
        - 'train_largest': Show the largest train categories.
        - 'test_largest': Show the largest test categories.
        - 'largest_difference': Show the largest difference between categories.
    classes_to_display : Optional[List[float]], default: None
        List of classes to display. The distribution of the properties would include only samples belonging (or
        containing an annotation belonging) to one of these classes. If None, samples from all classes are displayed.
    min_samples: int, default: 10
        Minimum number of samples needed in each dataset needed to calculate the drift.
    max_num_categories: int, default: None
        Deprecated. Please use max_num_categories_for_drift and max_num_categories_for_display instead
    """

    _IMAGE_THUMBNAIL_SIZE = (200, 200)

    def __init__(
            self,
            image_properties: t.Optional[t.List[t.Dict[str, t.Any]]] = None,
            margin_quantile_filter: float = 0.025,
            max_num_categories_for_drift: int = 10,
            max_num_categories_for_display: int = 10,
            show_categories_by: str = 'largest_difference',
            classes_to_display: t.Optional[t.List[str]] = None,
            min_samples: int = 30,
            max_num_categories: t.Optional[int] = None,  # Deprecated
            **kwargs
    ):
        super().__init__(**kwargs)
        if image_properties is not None:
            validate_properties(image_properties)
            self.image_properties = image_properties
        else:
            self.image_properties = default_image_properties

        self.margin_quantile_filter = margin_quantile_filter
        if max_num_categories is not None:
            warnings.warn(
                f'{self.__class__.__name__}: max_num_categories is deprecated. '
                'Please use "max_num_categories_for_drift" '
                'and "max_num_categories_for_display" instead',
                DeprecationWarning
            )
            max_num_categories_for_drift = max_num_categories_for_drift or max_num_categories
            max_num_categories_for_display = max_num_categories_for_display or max_num_categories
        self.max_num_categories_for_drift = max_num_categories_for_drift
        self.max_num_categories_for_display = max_num_categories_for_display
        self.show_categories_by = show_categories_by
        self.classes_to_display = set(classes_to_display) if classes_to_display else None
        self.min_samples = min_samples
        self._train_properties = None
        self._test_properties = None
        self._class_to_string = None

    def initialize_run(self, context: Context):
        """Initialize self state, and validate the run context."""
        self._train_properties = defaultdict(list)
        self._test_properties = defaultdict(list)

    def update(
        self,
        context: Context,
        batch: Batch,
        dataset_kind: DatasetKind
    ):
        """Calculate image properties for train or test batch."""
        if dataset_kind == DatasetKind.TRAIN:
            assert self._train_properties is not None
            properties = self._train_properties
        elif dataset_kind == DatasetKind.TEST:
            assert self._test_properties is not None
            properties = self._test_properties
        else:
            raise RuntimeError(
                f'Internal Error - Should not reach here! unknown dataset_kind: {dataset_kind}'
            )

        images = batch.images
        labels = batch.labels
        dataset = context.get_data_by_kind(dataset_kind)

        if self.classes_to_display:
            # use only images belonging (or containing an annotation belonging) to one of the classes in
            # classes_to_display
            class_to_string = dataset.label_id_to_name
            images_classes = dataset.get_classes(labels)

            # Iterator[tuple[image-index, set[image-classes]]]
            images_classes = (
                (index, set(map(class_to_string, image_classes)))
                for index, image_classes in enumerate(images_classes)
            )
            images = [
                images[index]
                for index, classes in images_classes
                if len(classes & self.classes_to_display) > 0
            ]

        for single_property in self.image_properties:
            calculated_properties = single_property['method'](images)
            properties[single_property['name']].extend(calculated_properties)

    def compute(self, context: Context) -> CheckResult:
        """Calculate drift score between train and test datasets for the collected image properties.

        Returns
        -------
        CheckResult
            value: dictionary containing drift score for each image property.
            display: distribution graph for each image property.
        """
        assert self._train_properties is not None
        assert self._test_properties is not None

        if sorted(self._train_properties.keys()) != sorted(self._test_properties.keys()):
            raise RuntimeError('Internal Error! Vision check was used improperly.')

        # if self.classes_to_display is set, check that it has classes that actually exist
        if self.classes_to_display is not None:
            class_to_string = context.train.label_id_to_name
            train_classes = set(map(class_to_string, context.train.classes_indices.keys()))
            if not self.classes_to_display.issubset(train_classes):
                raise DeepchecksValueError(
                    'Provided list of class ids to display '
                    f'{list(self.classes_to_display)} not found in training dataset.'
                )

        df_train = pd.DataFrame(self._train_properties)
        df_test = pd.DataFrame(self._test_properties)

        if len(df_train) < self.min_samples or len(df_test) < self.min_samples:
            raise NotEnoughSamplesError(
                f'Not enough samples to calculate drift score, minimum {self.min_samples} samples required'
                f', but got {len(df_train)} and {len(df_test)} samples in the train and test datasets.'
                'Use \'min_samples\' parameter to change the requirement.'
            )

        properties = sorted(self._train_properties.keys())
        figures = {}
        drifts = {}
        not_enough_samples = []

        for single_property in self.image_properties:
            property_name = single_property['name']

            try:
                score, _, figure = calc_drift_and_plot(
                    train_column=df_train[property_name],
                    test_column=df_test[property_name],
                    value_name=property_name,
                    column_type=get_column_type(single_property['output_type']),
                    margin_quantile_filter=self.margin_quantile_filter,
                    max_num_categories_for_drift=self.max_num_categories_for_drift,
                    max_num_categories_for_display=self.max_num_categories_for_display,
                    show_categories_by=self.show_categories_by,
                    min_samples=self.min_samples
                )
                figures[property_name] = figure
                drifts[property_name] = score
            except NotEnoughSamplesError:
                not_enough_samples.append(property_name)

        if len(drifts) == 0:
            drifts = None
            displays = []
        else:
            columns_order = sorted(properties, key=lambda col: drifts.get(col, 0), reverse=True)
            properties_to_display = [p for p in properties if p in drifts]
            # columns_order = sorted(properties, key=lambda col: drifts[col], reverse=True) # old

            headnote = (
                '<span>'
                'The Drift score is a measure for the difference between two distributions. '
                'In this check, drift is measured '
                f'for the distribution of the following image properties: {properties_to_display}.<br>'
                '</span> {additional}'
            )

            if not_enough_samples:
                headnote += (
                    f'<span>The following image properties do not have enough samples to calculate drift '
                    f'score: {not_enough_samples}</span>'
                )

            # out
            train_samples = df_train.sample(10)
            test_samples = df_test.sample(10)
            train_thumbnail_images, *_ = context.train.sample(*list(train_samples.index))
            test_thumbnail_images, *_ = context.test.sample(*list(train_samples.index))

            thumbnails = self._prepare_thumbnails_block(
                train_properties=train_samples.T,
                test_properties=test_samples.T,
                train_images=train_thumbnail_images,
                test_images=test_thumbnail_images
            )
            displays = [
                headnote,
                *[figures[col] for col in columns_order if col in figures],
                thumbnails
            ]

        return CheckResult(
            value=drifts,
            display=displays,
            header='Image Property Drift'
        )

    def _prepare_thumbnails_block(
        self,
        train_properties: pd.DataFrame,
        test_properties: pd.DataFrame,
        train_images: t.Sequence[pilimage.Image],
        test_images: t.Sequence[pilimage.Image]
    ) -> str:
        thumbnail_size = self._IMAGE_THUMBNAIL_SIZE
        tables = []

        for images, properties in ((train_images, train_properties), (test_images, test_properties),):
            properties_rows = []
            for name, values in properties.iterrows():
                properties_rows.append(f'<h4>{name}</h4>')
                for v in values:
                    properties_rows.append(f'<h4>{format_number(v)}</h4>')

            properties = ''.join(properties_rows)
            thumbnails = ''.join([
                prepare_thumbnail(img, size=thumbnail_size)
                for img in images
            ])

            tables.append(prepare_grid(
                content=f'<h4>Image</h4>{thumbnails}{properties}',
                style={
                    'grid-template-rows': 'auto 1fr 1fr',
                    'grid-template-columns': f'auto repeat({len(images)}, 1fr)'}
            ))

        train_table, test_table = tables

        template = dedent("""
        <h4>Train Images</h3>
        <hr>
        {train_thumbnails}
        <h4>Test Images</h4>
        <hr>
        {test_thumbnails}
        """)

        return template.format(
            train_thumbnails=train_table,
            test_thumbnails=test_table,
        )

    def add_condition_drift_score_not_greater_than(
        self: TImagePropertyDrift,
        max_allowed_drift_score: float = 0.1
    ) -> TImagePropertyDrift:
        """
        Add condition - require drift score to not be more than a certain threshold.

        Parameters
        ----------
        max_allowed_drift_score: float ,  default: 0.1
            the max threshold for the Earth Mover's Distance score

        Returns
        -------
        ConditionResult
            False if any column has passed the max threshold, True otherwise
        """

        def condition(result: t.Dict[str, float]) -> ConditionResult:
            failed_properties = [
                (property_name, drift_score)
                for property_name, drift_score in result.items()
                if drift_score > max_allowed_drift_score
            ]
            if len(failed_properties) > 0:
                failed_properties = ';\n'.join(f'{p}={d:.2f}' for p, d in failed_properties)
                return ConditionResult(
                    ConditionCategory.FAIL,
                    'Earth Mover\'s Distance is above the threshold '
                    f'for the next properties:\n{failed_properties}'
                )
            return ConditionResult(ConditionCategory.PASS)

        return self.add_condition(
            f'Earth Mover\'s Distance <= {max_allowed_drift_score} for image properties drift',
            condition
        )
