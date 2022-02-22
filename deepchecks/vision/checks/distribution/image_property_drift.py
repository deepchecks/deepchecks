import typing as t
from numbers import Number
from collections import defaultdict

import pandas as pd

from deepchecks.core import DatasetKind
from deepchecks.core import CheckResult
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.vision import TrainTestCheck
from deepchecks.vision import Context
from deepchecks.vision.utils import ImageFormatter
from deepchecks.utils.distribution.drift import calc_drift_and_plot


__all__ = ['ImagePropertyDrift',]


ImageProperty = t.Union[str, t.Callable[..., t.List[Number]]]


class ImagePropertyDrift(TrainTestCheck):
    """
    Calculate drift between train dataset and test dataset per image property, using statistical measures.

    Check calculates a drift score for each image property in test dataset, by comparing its distribution to the train
    dataset. For this, we use the Earth Movers Distance.

    See https://en.wikipedia.org/wiki/Wasserstein_metric

    Pramaters
    ---------
    image_properties : Optional[List[Union[str, Callable[..., Number]]]]
    """

    def __init__(
        self,
        image_properties: t.Optional[t.List[ImageProperty]] = None,
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

        self.train_properties = defaultdict(list)
        self.test_properties = defaultdict(list)

    def update(
        self,
        context: Context,
        batch: t.Any,
        dataset_kind: DatasetKind
    ):
        """Calculate image properties for train or test batches."""
        if dataset_kind == DatasetKind.TRAIN:
            dataset = context.train
            properties = self.train_properties
        elif dataset_kind == DatasetKind.TEST:
            dataset = context.test
            properties = self.test_properties
        else:
            raise RuntimeError(
                "Internal Error! Part of code that must "
                "be unreacheable was reached."
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
            raise DeepchecksValueError('') # TODO: message

        properties = sorted(self.train_properties.keys())
        df_train = pd.DataFrame(self.train_properties)
        df_test = pd.DataFrame(self.test_properties)

        figures = []
        drifts = {}

        for property_name in properties:
            score, _, figure = calc_drift_and_plot(
                train_column=df_train[property_name],
                test_column=df_test[property_name],
                column_type='numerical',
                plot_title=property_name
            )
            figures.append(figure)
            drifts[property_name] = {'Drift score': score,}

        headnote = '' # TODO:

        return CheckResult(
            drifts,
            display=[headnote, *figures],
            header='Image Property Drift'
        )
