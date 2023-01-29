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
"""Module for custom scorer metric."""
import typing as t

import numpy as np
import pandas as pd
from ignite.metrics import Metric
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce

from deepchecks.tabular import Dataset
from deepchecks.tabular.context import _DummyModel
from deepchecks.tabular.metric_utils import DeepcheckScorer
from deepchecks.vision.vision_data.utils import object_to_numpy


class CustomClassificationScorer(Metric):
    """Scorer that runs a custom metric for the vision classification task.

    Custom scorers can be passed to all model evaluation related checks as can be seen in the example below.

    Parameters
    ----------
    scorer : t.Union[str, t.Callable]
        sklearn scorer name or deepchecks supported string o rcallable

    Returns
    -------
    scorer: DeepcheckScorer
        An initialized DeepcheckScorer.

    Examples
    --------
    >>> from sklearn.metrics import make_scorer, cohen_kappa_score
    ... from deepchecks.vision.metrics_utils.custom_scorer import CustomClassificationScorer
    ... from deepchecks.vision.checks.model_evaluation import SingleDatasetPerformance
    ... from deepchecks.vision.datasets.classification import mnist_torch as mnist
    ...
    ... mnist_model = mnist.load_model()
    ... test_ds = mnist.load_dataset(root='Data', object_type='VisionData')
    ...
    >>> ck = CustomClassificationScorer(make_scorer(cohen_kappa_score))
    ...
    >>> check = SingleDatasetPerformance(scorers={'cohen_kappa_score': ck})
    ... check.run(test_ds, mnist_model).value
    """

    def __init__(
            self,
            scorer: t.Union[t.Callable, str],
    ):
        super().__init__(device="cpu")
        self.scorer = scorer

    @reinit__is_reduced
    def reset(self):
        """Reset metric state."""
        self._y_proba = []
        self._y = []

        super().reset()

    @reinit__is_reduced
    def update(self, output):
        """Update metric with batch of samples."""
        y_proba, y = output
        self._y_proba.append(object_to_numpy(y_proba))
        self._y.append(object_to_numpy(y))

    @sync_all_reduce("_y_proba", "_y")
    def compute(self):
        """Compute metric value."""
        y_proba = np.concatenate(self._y_proba)
        y = np.concatenate(self._y)

        classes = list(range(y_proba.shape[1]))

        dummy_dataset = Dataset(df=pd.DataFrame(y_proba), label=y, cat_features=[])
        dummy_model = _DummyModel(test=dummy_dataset, y_proba_test=y_proba, model_classes=classes)

        deep_checks_scorer = DeepcheckScorer(self.scorer, model_classes=classes, observed_classes=classes)
        return deep_checks_scorer(dummy_model, dummy_dataset)
