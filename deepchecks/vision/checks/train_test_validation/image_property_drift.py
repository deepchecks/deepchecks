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

import pandas as pd

from deepchecks.core import CheckResult, ConditionResult, DatasetKind
from deepchecks.core.condition import ConditionCategory
from deepchecks.core.errors import DeepchecksValueError, NotEnoughSamplesError
from deepchecks.utils.dict_funcs import get_dict_entry_by_value
from deepchecks.utils.distribution.drift import calc_drift_and_plot
from deepchecks.utils.strings import format_number
from deepchecks.vision import Batch, Context, TrainTestCheck
from deepchecks.vision.utils.image_properties import default_image_properties, get_column_type
from deepchecks.vision.utils.vision_properties import PropertiesInputType

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
        representing attributes of said method. 'output_type' must be one of:
        - 'numeric' - for continuous ordinal outputs.
        - 'categorical' - for discrete, non-ordinal outputs. These can still be numbers,
          but these numbers do not have inherent value.
        For more on image / label properties, see the :ref:`property guide </user-guide/vision/vision_properties.rst>`
    margin_quantile_filter: float, default: 0.025
        float in range [0,0.5), representing which margins (high and low quantiles) of the distribution will be filtered
        out of the EMD calculation. This is done in order for extreme values not to affect the calculation
        disproportionally. This filter is applied to both distributions, in both margins.
    max_num_categories_for_drift: int, default: 10
        Only for discrete properties. Max number of allowed categories. If there are more,
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
    min_samples: int, default: 30
        Minimum number of samples needed in each dataset needed to calculate the drift.
    max_num_categories: int, default: None
        Deprecated. Please use max_num_categories_for_drift and max_num_categories_for_display instead
    """

    def __init__(
            self,
            image_properties: t.List[t.Dict[str, t.Any]] = None,
            margin_quantile_filter: float = 0.025,
            max_num_categories_for_drift: int = 10,
            max_num_categories_for_display: int = 10,
            show_categories_by: str = 'largest_difference',
            classes_to_display: t.Optional[t.List[str]] = None,
            min_samples: int = 30,
            max_num_categories: int = None,  # Deprecated
            **kwargs
    ):
        super().__init__(**kwargs)
        self.image_properties = image_properties if image_properties else default_image_properties
        self.margin_quantile_filter = margin_quantile_filter
        if max_num_categories is not None:
            warnings.warn(
                f'{self.__class__.__name__}: max_num_categories is deprecated. please use max_num_categories_for_drift '
                'and max_num_categories_for_display instead',
                DeprecationWarning
            )
            max_num_categories_for_drift = max_num_categories_for_drift or max_num_categories
            max_num_categories_for_display = max_num_categories_for_display or max_num_categories
        self.max_num_categories_for_drift = max_num_categories_for_drift
        self.max_num_categories_for_display = max_num_categories_for_display
        self.show_categories_by = show_categories_by
        self.classes_to_display = classes_to_display
        self.min_samples = min_samples

        self._train_properties = None
        self._test_properties = None
        self._class_to_string = None

    def initialize_run(self, context: Context):
        """Initialize self state, and validate the run context."""
        self._class_to_string = context.train.label_id_to_name

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
            properties_results = self._train_properties
        elif dataset_kind == DatasetKind.TEST:
            properties_results = self._test_properties
        else:
            raise RuntimeError(
                f'Internal Error - Should not reach here! unknown dataset_kind: {dataset_kind}'
            )

        all_classes_properties = batch.vision_properties(
            batch.images, self.image_properties, PropertiesInputType.IMAGES)

        if self.classes_to_display:
            # use only images belonging (or containing an annotation belonging) to one of the classes in
            # classes_to_display
            classes = context.train.get_classes(batch.labels)
            filtered_properties = dict.fromkeys(all_classes_properties.keys())
            for prop_name, prop_values in all_classes_properties.items():
                filtered_properties[prop_name] = [score for idx, score in enumerate(prop_values)
                                                  if any(cls in map(self._class_to_string, classes[idx]) for cls
                                                         in self.classes_to_display)]
        else:
            filtered_properties = all_classes_properties

        for prop_name, property_values in filtered_properties.items():
            properties_results[prop_name].extend(property_values)

    def compute(self, context: Context) -> CheckResult:
        """Calculate drift score between train and test datasets for the collected image properties.

        Returns
        -------
        CheckResult
            value: dictionary containing drift score for each image property.
            display: distribution graph for each image property.
        """
        if sorted(self._train_properties.keys()) != sorted(self._test_properties.keys()):
            raise RuntimeError('Internal Error! Vision check was used improperly.')

        # if self.classes_to_display is set, check that it has classes that actually exist
        if self.classes_to_display is not None:
            if not set(self.classes_to_display).issubset(
                    map(self._class_to_string, context.train.classes_indices.keys())
            ):
                raise DeepchecksValueError(
                    f'Provided list of class ids to display {self.classes_to_display} not found in training dataset.'
                )

        properties = sorted(self._train_properties.keys())
        df_train = pd.DataFrame(self._train_properties)
        df_test = pd.DataFrame(self._test_properties)
        if len(df_train) < self.min_samples or len(df_test) < self.min_samples:
            raise NotEnoughSamplesError(
                f'Not enough samples to calculate drift score, minimum {self.min_samples} samples required'
                f', but got {len(df_train)} and {len(df_test)} samples in the train and test datasets.'
                'Use \'min_samples\' parameter to change the requirement.'
            )

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
                    min_samples=self.min_samples,
                    with_display=context.with_display,
                )

                figures[property_name] = figure
                drifts[property_name] = score
            except NotEnoughSamplesError:
                not_enough_samples.append(property_name)

        if context.with_display:
            columns_order = sorted(properties, key=lambda col: drifts.get(col, 0), reverse=True)
            properties_to_display = [p for p in properties if p in drifts]

            headnote = '<span>' \
                       'The Drift score is a measure for the difference between two distributions. ' \
                       'In this check, drift is measured ' \
                       f'for the distribution of the following image properties: {properties_to_display}.<br>' \
                       '</span>'
            if not_enough_samples:
                headnote += f'<span>The following image properties do not have enough samples to calculate drift ' \
                            f'score: {not_enough_samples}</span>'

            displays = [headnote] + [figures[col] for col in columns_order if col in figures]
        else:
            displays = []

        return CheckResult(
            value=drifts if drifts else {},
            display=displays,
            header='Image Property Drift'
        )

    def add_condition_drift_score_less_than(
        self: TImagePropertyDrift,
        max_allowed_drift_score: float = 0.1
    ) -> TImagePropertyDrift:
        """
        Add condition - require drift score to be less than a certain threshold.

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
                if drift_score >= max_allowed_drift_score
            ]
            if len(failed_properties) > 0:
                failed_properties = ';\n'.join(f'{p}={format_number(d)}' for p, d in failed_properties)
                return ConditionResult(
                    ConditionCategory.FAIL,
                    'Earth Mover\'s Distance is above the threshold '
                    f'for the next properties:\n{failed_properties}'
                )
            else:
                if not result:
                    details = 'Did not calculate drift score on any property'
                else:
                    prop, score = get_dict_entry_by_value(result)
                    details = f'Found property {prop} with largest Earth Mover\'s Distance score {format_number(score)}'
                return ConditionResult(ConditionCategory.PASS, details)

        return self.add_condition(
            f'Earth Mover\'s Distance < {max_allowed_drift_score} for image properties drift',
            condition
        )
