# ----------------------------------------------------------------------------
# Copyright (C) 2021-2023 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""Utils module containing functionalities to infer the metadata from the supplied TextData."""

import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
from seqeval.metrics.sequence_labeling import get_entities
from sklearn.base import BaseEstimator

from deepchecks.core.errors import DeepchecksValueError
from deepchecks.nlp.task_type import TaskType

__all__ = ['infer_observed_and_model_labels']


# pylint: disable=unused-argument
def infer_observed_and_model_labels(
    train_dataset=None,
    test_dataset=None,
    model: BaseEstimator = None,
    y_pred_train: np.ndarray = None,
    y_pred_test: np.ndarray = None,
    model_classes: list = None,
    task_type: TaskType = None
) -> Tuple[List, List]:
    """
    Infer the observed labels from the given datasets and predictions.

    Parameters
    ----------
    train_dataset : Union[TextData, None], default None
        TextData object, representing data an estimator was fitted on
    test_dataset : Union[TextData, None], default None
        TextData object, representing data an estimator predicts on
    model : Union[BaseEstimator, None], default None
        A fitted estimator instance
    y_pred_train : np.array
        Predictions on train_dataset
    y_pred_test : np.array
        Predictions on test_dataset
    model_classes : Optional[List], default None
        list of classes known to the model
    task_type : Union[TaskType, None], default None
        The task type of the model

    Returns
    -------
        observed_classes : list
            List of observed label values. For multi-label, returns number of observed labels.
        model_classes : list
            List of the user-given model classes. For multi-label, if not given by the user, returns a range of
            len(label)
    """
    # TODO: Doesn't work for predictions
    train_labels = []
    test_labels = []
    have_model = model is not None  # Currently irrelevant as no model is given in NLP

    if train_dataset:
        if train_dataset.has_label():
            train_labels += list(train_dataset.label)
        if have_model:
            train_labels += list(model.predict(train_dataset))
    if test_dataset:
        if test_dataset.has_label():
            test_labels += list(test_dataset.label)
        if have_model:
            test_labels += list(model.predict(test_dataset))

    if task_type == TaskType.TOKEN_CLASSIFICATION:
        # Flatten:
        train_labels = [token_label for sentence in train_labels for token_label in sentence]
        test_labels = [token_label for sentence in test_labels for token_label in sentence]

        if model_classes and 'O' in model_classes:
            model_classes = [c for c in model_classes if c != 'O']
            warnings.warn(
                '"O" label was removed from model_classes as it is ignored by metrics for token classification',
                UserWarning)

    observed_classes = np.array(test_labels + train_labels, dtype=object)
    if len(observed_classes.shape) == 2:  # For the multi-label case
        len_observed_label = observed_classes.shape[1]
        if not model_classes:
            model_classes = list(range(len_observed_label))
            observed_classes = list(range(len_observed_label))
        else:
            if len(model_classes) != len_observed_label:
                raise DeepchecksValueError(f'Received model_classes of length {len(model_classes)}, '
                                           f'but data indicates labels of length {len_observed_label}')
            observed_classes = model_classes
    else:
        observed_classes = observed_classes[~pd.isnull(observed_classes)]
        observed_classes = sorted(np.unique(observed_classes))

    if task_type == TaskType.TOKEN_CLASSIFICATION:
        observed_classes = [c for c in observed_classes if c != 'O']
        observed_classes = sorted({tag for tag, _, _ in get_entities(observed_classes)})

    return observed_classes, model_classes
