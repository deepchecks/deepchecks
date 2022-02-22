import typing as t
from numbers import Number
from collections import defaultdict

import torch
import pandas as pd

from deepchecks.core import DatasetKind
from deepchecks.core import CheckResult
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.vision import TrainTestCheck
from deepchecks.vision import Context
from deepchecks.vision.utils import ImageFormatter
from deepchecks.utils.distribution.drift import calc_drift_and_plot


__all__ = ['ImagePropertyDrift',]


ImageProperty = t.Union[str, t.Callable[..., Number]]

class ImagePropertyDrift(TrainTestCheck):
    """
    Calculate drift between train dataset and test dataset per feature, using statistical measures.

    Check calculates a drift score for each column in test dataset, by comparing its distribution to the train
    dataset.
    For numerical columns, we use the Earth Movers Distance.
    See https://en.wikipedia.org/wiki/Wasserstein_metric
    For categorical columns, we use the Population Stability Index (PSI).
    See https://www.lexjansen.com/wuss/2017/47_Final_Paper_PDF.pdf
    """
    
    def __init__(
        self,
        image_properties: t.Optional[t.List[ImageProperty]] = None,
        sample_size: int = 10_000,
        random_state: int = 42,
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

        self.random_state = random_state
        self.sample_size = sample_size
        
        self.train_properties = defaultdict(list)
        self.test_properties = defaultdict(list)
    
    def update(
        self, 
        context: Context,
        batch: t.Any, 
        dataset_kind: DatasetKind
    ):
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

        for name in ImageFormatter.IMAGE_PROPERTIES:
            properties[name].extend(
                getattr(dataset.image_formatter, name)(images)
            )

    def compute(self, context: Context) -> CheckResult:
        df_train = pd.DataFrame(self.train_properties)
        df_test = pd.DataFrame(self.test_properties)

        figures = []
        drifts = {}

        for property_name in sorted(ImageFormatter.IMAGE_PROPERTIES):
            score, method, figure = calc_drift_and_plot(
                train_column=df_train[property_name],
                test_column=df_test[property_name],
                column_type='numerical',
                plot_title=property_name
            )
            figures.append(figure)
            drifts[property_name] = {'Drift score': score, 'Method': method} 
        
        headnote = "" # TODO:

        return CheckResult(
            drifts, 
            display=[headnote, *figures],
            header="Image Property Drift"
        )
