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
from collections import defaultdict

import pandas as pd

from deepchecks.core import CheckResult, ConditionResult, DatasetKind
from deepchecks.core.condition import ConditionCategory
from deepchecks.core.errors import DeepchecksValueError, NotEnoughSamplesError
from deepchecks.core.reduce_classes import ReducePropertyMixin
from deepchecks.utils.dict_funcs import get_dict_entry_by_value
from deepchecks.utils.distribution.drift import calc_drift_and_plot, get_drift_plot_sidenote
from deepchecks.utils.strings import format_number
from deepchecks.vision._shared_docs import docstrings
from deepchecks.vision.base_checks import TrainTestCheck
from deepchecks.vision.context import Context
from deepchecks.vision.utils.image_properties import default_image_properties
from deepchecks.vision.utils.vision_properties import PropertiesInputType
from deepchecks.vision.vision_data.batch_wrapper import BatchWrapper

__all__ = ['ImagePropertyDrift']

TImagePropertyDrift = t.TypeVar('TImagePropertyDrift', bound='ImagePropertyDrift')


@docstrings
class ImagePropertyDrift(TrainTestCheck, ReducePropertyMixin):
    """
    Calculate drift between train dataset and test dataset per image property, using statistical measures.

    Check calculates a drift score for each image property in test dataset, by comparing its distribution to the train
    dataset. For this, we use the Earth Movers Distance.

    See https://en.wikipedia.org/wiki/Wasserstein_metric

    Parameters
    ----------
    image_properties : List[Dict[str, Any]], default: None
        List of properties. Replaces the default deepchecks properties.
        Each property is a dictionary with keys ``'name'`` (str), ``method`` (Callable) and ``'output_type'`` (str),
        representing attributes of said method. 'output_type' must be one of:

        - ``'numeric'`` - for continuous ordinal outputs.
        - ``'categorical'`` - for discrete, non-ordinal outputs. These can still be numbers,
          but these numbers do not have inherent value.

        For more on image / label properties, see the guide about :ref:`vision_properties_guide`.
    margin_quantile_filter: float, default: 0.025
        float in range [0,0.5), representing which margins (high and low quantiles) of the distribution will be filtered
        out of the EMD calculation. This is done in order for extreme values not to affect the calculation
        disproportionally. This filter is applied to both distributions, in both margins.
    min_category_size_ratio: float, default 0.01
        minimum size ratio for categories. Categories with size ratio lower than this number are binned
        into an "Other" category.
    max_num_categories_for_drift: int, default: None
        Only for discrete properties. Max number of allowed categories. If there are more,
        they are binned into an "Other" category. This limit applies for both drift calculation and distribution plots.
    max_num_categories_for_display: int, default: 10
        Max number of categories to show in plot.
    show_categories_by: str, default: 'largest_difference'
        Specify which categories to show for categorical features' graphs, as the number of shown categories is limited
        by max_num_categories_for_display. Possible values:

        - 'train_largest': Show the largest train categories.
        - 'test_largest': Show the largest test categories.
        - 'largest_difference': Show the largest difference between categories.

    min_samples: int, default: 30
        Minimum number of samples needed in each dataset needed to calculate the drift.
    aggregation_method: str, default: 'max'
        {property_aggregation_method_argument:2*indent}
    {additional_check_init_params:2*indent}
    """

    def __init__(
            self,
            image_properties: t.List[t.Dict[str, t.Any]] = None,
            margin_quantile_filter: float = 0.025,
            max_num_categories_for_drift: int = None,
            min_category_size_ratio: float = 0.01,
            max_num_categories_for_display: int = 10,
            show_categories_by: str = 'largest_difference',
            min_samples: int = 30,
            aggregation_method: str = 'max',
            n_samples: t.Optional[int] = 10000,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.n_samples = n_samples
        self.image_properties = image_properties
        self.margin_quantile_filter = margin_quantile_filter
        self.max_num_categories_for_drift = max_num_categories_for_drift
        self.min_category_size_ratio = min_category_size_ratio
        self.max_num_categories_for_display = max_num_categories_for_display
        self.show_categories_by = show_categories_by
        self.min_samples = min_samples
        self.aggregation_method = aggregation_method

        self._train_properties = None
        self._test_properties = None

    def initialize_run(self, context: Context):
        """Initialize self state, and validate the run context."""
        self._train_properties = defaultdict(list)
        self._test_properties = defaultdict(list)

    def update(
            self,
            context: Context,
            batch: BatchWrapper,
            dataset_kind: DatasetKind
    ):
        """Calculate image properties for train or test batch."""
        if dataset_kind == DatasetKind.TRAIN:
            properties_results = self._train_properties
        elif dataset_kind == DatasetKind.TEST:
            properties_results = self._test_properties
        else:
            raise DeepchecksValueError(f'Invalid dataset kind: {dataset_kind}')

        all_classes_properties = batch.vision_properties(self.image_properties, PropertiesInputType.IMAGES)
        for prop_name, property_values in all_classes_properties.items():
            properties_results[prop_name].extend(property_values)

    def compute(self, context: Context) -> CheckResult:
        """Calculate drift score between train and test datasets for the collected image properties.

        Returns
        -------
        CheckResult
            value: dictionary containing drift score for each image property.
            display: distribution graph for each image property.
        """
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

        dataset_names = (context.train.name, context.test.name)

        for single_property in self.image_properties or default_image_properties:
            property_name = single_property['name']
            if property_name not in df_train.columns or property_name not in df_test.columns:
                continue
            try:
                score, _, figure = calc_drift_and_plot(
                    train_column=df_train[property_name],
                    test_column=df_test[property_name],
                    value_name=property_name,
                    column_type=single_property['output_type'],
                    margin_quantile_filter=self.margin_quantile_filter,
                    max_num_categories_for_drift=self.max_num_categories_for_drift,
                    min_category_size_ratio=self.min_category_size_ratio,
                    max_num_categories_for_display=self.max_num_categories_for_display,
                    show_categories_by=self.show_categories_by,
                    min_samples=self.min_samples,
                    with_display=context.with_display,
                    dataset_names=dataset_names
                )

                figures[property_name] = figure
                drifts[property_name] = score
            except NotEnoughSamplesError:
                not_enough_samples.append(property_name)

        if context.with_display:
            columns_order = sorted(properties, key=lambda col: drifts.get(col, 0), reverse=True)
            properties_to_display = [p for p in properties if p in drifts]

            headnote = ['<span> The Drift score is a measure for the difference between two distributions. '
                        'In this check, drift is measured for the distribution '
                        f'of the following image properties: {properties_to_display}.</span>',
                        get_drift_plot_sidenote(self.max_num_categories_for_display, self.show_categories_by)]

            if not_enough_samples:
                headnote.append(f'<span>The following image properties do not have enough samples to calculate drift '
                                f'score: {not_enough_samples}</span>')

            displays = headnote + [figures[col] for col in columns_order if col in figures]
        else:
            displays = []

        return CheckResult(
            value=drifts if drifts else {},
            display=displays,
            header='Image Property Drift'
        )

    def reduce_output(self, check_result: CheckResult) -> t.Dict[str, float]:
        """Return prediction drift score per prediction property."""
        return self.property_reduce(self.aggregation_method, pd.Series(check_result.value), 'Drift Score')

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
