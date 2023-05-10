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
"""Utils module containing utilities for nlp checks working with scorers."""

import typing as t

import numpy as np

from deepchecks.nlp.task_type import TaskType
from deepchecks.nlp.text_data import TextData
from deepchecks.tabular.metric_utils import DeepcheckScorer
from deepchecks.tabular.metric_utils.scorers import _transform_to_multi_label_format
from deepchecks.utils.metrics import is_label_none
from deepchecks.utils.typing import ClassificationModel

__all__ = [
    'init_validate_scorers',
    'infer_on_text_data'
]


def init_validate_scorers(scorers: t.Union[t.Mapping[str, t.Union[str, t.Callable]], t.List[str]],
                          model_classes: t.Optional[t.List],
                          observed_classes: t.Optional[t.List]) -> t.List[DeepcheckScorer]:
    """Initialize scorers and return all of them as deepchecks scorers.

    Parameters
    ----------
    scorers : Mapping[str, Union[str, Callable]]
        dict of scorers names to scorer sklearn_name/function or a list without a name
    model_classes : t.Optional[t.List]
        possible classes output for model. None for regression tasks.
    observed_classes : t.Optional[t.List]
        observed classes from labels and predictions. None for regression tasks.

    Returns
    -------
    t.List[DeepcheckScorer]
        A list of initialized scorers
    """
    if isinstance(scorers, t.Mapping):
        scorers: t.List[DeepcheckScorer] = [DeepcheckScorer(scorer, model_classes, observed_classes, name)
                                            for name, scorer in scorers.items()]
    else:
        scorers: t.List[DeepcheckScorer] = [DeepcheckScorer(scorer, model_classes, observed_classes)
                                            for scorer in scorers]
    return scorers


def infer_on_text_data(scorer: DeepcheckScorer, model: ClassificationModel, data: TextData, drop_na: bool = True):
    """Infer using DeepcheckScorer on NLP TextData using an NLP context _DummyModel."""
    y_pred = model.predict(data)
    y_true = data.label

    if drop_na:
        idx_to_keep = [not(is_label_none(pred) or is_label_none(label)) for pred, label in zip(y_pred, y_true)]
        y_pred = np.asarray(y_pred, dtype='object')[idx_to_keep]
        y_true = y_true[idx_to_keep]

    if data.task_type == TaskType.TEXT_CLASSIFICATION:
        y_pred = _transform_to_multi_label_format(y_pred, scorer.model_classes).astype(int)
        y_true = _transform_to_multi_label_format(y_true, scorer.model_classes).astype(int)

    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(data)
        if drop_na and y_proba is not None:
            y_proba = np.asarray(y_proba, 'object')[idx_to_keep].astype(float)
    else:
        y_proba = None
    results = scorer.run_on_pred(y_true, y_pred, y_proba)
    return scorer.validate_scorer_multilabel_output(results)
