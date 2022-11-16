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
"""Utils module containing functionalities to infer the metadata from the supplied TextData."""

__all__ = ['infer_observed_labels']

from typing import List, Union

import numpy as np
import pandas as pd


def infer_observed_labels(train_dataset=None, test_dataset=None, model=None) -> Union[List, List[List]]:
    """
    Infer the observed labels from the given datasets and predictions.

    Parameters
    ----------
    train_dataset: Union[TextData, None], default: None
        TextData object, representing data an estimator was fitted on
    test_dataset: Union[TextData, None], default: None
        TextData object, representing data an estimator predicts on
    model: Union[BaseModel, None], default: None
        A fitted estimator instance

    Returns
    -------
        Union[List, List[List]]:
            List of observed label values. For multilabel, returns list of lists of observed label values.
    """
    train_labels = []
    test_labels = []
    have_model = model is not None
    if train_dataset:
        if train_dataset.has_label():
            train_labels += train_dataset.label
        if have_model:
            train_labels += model.predict(train_dataset)
    if test_dataset:
        if test_dataset.has_label():
            test_labels += test_dataset.label
        if have_model:
            test_labels += model.predict(test_dataset)

    observed_labels = np.array(test_labels + train_labels)
    if len(observed_labels.shape) == 2:
        observed_labels = [np.unique(observed_labels[:, i]) for i in range(observed_labels.shape[1])]
        observed_labels = [x[~pd.isnull(x)] for x in observed_labels]
        observed_labels = [sorted(x) for x in observed_labels]
        # Bressler - this is more generalized for multilabel, instead of being specific for [0,1] labels. Should keep?
    else:
        observed_labels = np.unique(observed_labels)
        observed_labels = observed_labels[~pd.isnull(observed_labels)]
        observed_labels = sorted(observed_labels)
    return observed_labels
