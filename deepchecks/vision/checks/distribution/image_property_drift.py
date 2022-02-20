import typing as t
from collections import defaultdict

import torch
import pandas as pd

from deepchecks.core import DatasetKind
from deepchecks.core import CheckResult
from deepchecks.vision import TrainTestCheck
from deepchecks.vision import Context
from deepchecks.vision.utils import ImageFormatter
from deepchecks.utils.distribution.drift import calc_drift_and_plot


__all__ = ['ImagePropertyDrift',]


class ImagePropertyDrift(TrainTestCheck):
    
    def __init__(
        self,
        random_state: int = 42,
        test_size: float = 0.3
    ):
        super().__init__()
        self.random_state = random_state
        self.test_size = test_size
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

        images = dataset.image_transformer(batch[0])

        for name in ImageFormatter.IMAGE_PROPERTIES:
            properties[name].extend(
                getattr(dataset.image_transformer, name)(images)
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
