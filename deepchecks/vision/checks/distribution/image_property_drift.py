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
from numbers import Number
from collections import defaultdict

import numpy as np
import pandas as pd

from deepchecks.core import DatasetKind
from deepchecks.core import CheckResult
from deepchecks.core import ConditionResult
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.vision import TrainTestCheck
from deepchecks.vision import Context
from deepchecks.vision.utils import ImageFormatter

from .train_test_label_drift import calc_drift_and_plot


__all__ = ['ImagePropertyDrift']


ImageProperty = t.Union[str, t.Callable[..., t.List[Number]]]


TImagePropertyDrift = t.TypeVar('TImagePropertyDrift', bound='ImagePropertyDrift')


class ImagePropertyDrift(TrainTestCheck):
    """
    Calculate drift between train dataset and test dataset per image property, using statistical measures.

    Check calculates a drift score for each image property in test dataset, by comparing its distribution to the train
    dataset. For this, we use the Earth Movers Distance.

    See https://en.wikipedia.org/wiki/Wasserstein_metric

    Pramaters
    ---------
    image_properties : Optional[List[Union[str, Callable[..., Number]]]]
    default_number_of_bins: int, default: 100
    """

    def __init__(
        self,
        image_properties: t.Optional[t.List[ImageProperty]] = None,
        default_number_of_bins: int = 100
    ):
        super().__init__()

        if image_properties is None:
            self.image_properties = ImageFormatter.IMAGE_PROPERTIES
        else:
            if len(image_properties) == 0:
                raise DeepchecksValueError('image_properties list cannot be empty')

            received_properties = {p for p in image_properties if isinstance(p, str)}
            unknown_properties = received_properties.difference(ImageFormatter.IMAGE_PROPERTIES)

            if len(unknown_properties) > 0:
                raise DeepchecksValueError(
                    'receivedd list of unknown image properties '
                    f'- {sorted(unknown_properties)}'
                )

            self.image_properties = image_properties

        self.default_number_of_bins = default_number_of_bins
        self.train_properties = defaultdict(list)
        self.test_properties = defaultdict(list)

    def update(
        self,
        context: Context,
        batch: t.Any,
        dataset_kind: DatasetKind
    ):
        """Calculate image properties for train or test batch."""
        if dataset_kind == DatasetKind.TRAIN:
            dataset = context.train
            properties = self.train_properties
        elif dataset_kind == DatasetKind.TEST:
            dataset = context.test
            properties = self.test_properties
        else:
            raise RuntimeError(
                'Internal Error! Part of code that must '
                'be unreacheable was reached.'
            )

        images = dataset.image_formatter(batch)

        for image_property in self.image_properties:
            if isinstance(image_property, str):
                properties[image_property].extend(
                    getattr(dataset.image_formatter, image_property)(images)
                )
            elif callable(image_property):
                # TODO: if it is a lambda it will have a name - <lambda>, that is a problem/
                properties[image_property.__name__].extend(image_property(images))
            else:
                raise DeepchecksValueError(
                    'Do not know how to work with image'
                    f'property of type - {type(image_property).__name__}'
                )

    def compute(self, context: Context) -> CheckResult:
        """Calculate drift score between train and test datasets for the collected image properties.

        Returns
        -------
        CheckResult
            value: dictionary containing drift score for each image property.
            display: distribution graph for each image property.
        """
        if sorted(self.train_properties.keys()) != sorted(self.test_properties.keys()):
            raise RuntimeError('Internal Error! Vision check was used improperly.')

        properties = sorted(self.train_properties.keys())
        df_train = pd.DataFrame(self.train_properties)
        df_test = pd.DataFrame(self.test_properties)

        figures = []
        drifts = {}

        for property_name in properties:
            lower_bound = min(df_train[property_name].min(), df_test[property_name].min())
            upper_bound = max(df_train[property_name].max(), df_test[property_name].max())
            bounds = (lower_bound, upper_bound)

            train_hist, train_edges = np.histogram(
                df_train[property_name],
                range=bounds,
                bins=self.default_number_of_bins
            )
            test_hist, test_edges = np.histogram(
                df_test[property_name],
                range=bounds,
                bins=self.default_number_of_bins
            )

            train_histogram = dict(zip(train_edges, train_hist))
            test_histogram = dict(zip(test_edges, test_hist))

            score, _, figure = calc_drift_and_plot(
                train_distribution=train_histogram,
                test_distribution=test_histogram,
                column_type='numerical',
                plot_title=property_name
            )

            figures.append(figure)
            drifts[property_name] = {'Drift score': score}

        if drifts:
            value = pd.DataFrame(drifts).T
            headnote = ''  # TODO:
            display = [headnote, *figures]
        else:
            value = None
            display = []

        return CheckResult(
            value=value,
            display=display,
            header='Image Property Drift'
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

        def condition(result: pd.DataFrame) -> ConditionResult:
            failed_properties = [
                (property_name, drift_score)
                for property_name, drift_score in result.itertuples()
                if drift_score > max_allowed_drift_score
            ]
            if len(failed_properties) > 0:
                failed_properties = ';\n'.join(f'{p}={d:.2f}' for p, d in failed_properties)
                return ConditionResult(
                    False,
                    'Earth Mover\'s Distance is above the threshold '
                    f'for the next properties:\n{failed_properties}'
                )
            return ConditionResult(True)

        return self.add_condition(
            f'Earth Mover\'s Distance <= {max_allowed_drift_score} for image propertiesdrift',
            condition
        )
