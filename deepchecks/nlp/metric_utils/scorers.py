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
"""Utils module containing utilities for nlp checks working with scorers."""

import typing as t
from deepchecks.nlp.text_data import TextData
from deepchecks.tabular.metric_utils import DeepcheckScorer
from deepchecks.utils.typing import ClassificationModel

__all__ = [
    'infer_on_text_data',
    'infer_on_text_data'
]


def init_validate_scorers(scorers: t.Union[t.Mapping[str, t.Union[str, t.Callable]], t.List[str]]
                          ) -> t.List[DeepcheckScorer]:
    """Initialize scorers and return all of them as deepchecks scorers.

    Parameters
    ----------
    scorers : Mapping[str, Union[str, Callable]]
        dict of scorers names to scorer sklearn_name/function or a list without a name
    """
    if isinstance(scorers, t.Mapping):
        scorers: t.List[DeepcheckScorer] = [DeepcheckScorer(scorer, name) for name, scorer in scorers.items()]
    else:
        scorers: t.List[DeepcheckScorer] = [DeepcheckScorer(scorer) for scorer in scorers]
    return scorers


def infer_on_text_data(scorer: DeepcheckScorer, model: ClassificationModel, data: TextData):
    """Infer using DeepcheckScorer on nlp TextData using a nlp context _DummyModel"""
    y_true = data.label
    y_pred = model.predict(data)
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(data)
    else:
        y_proba = None
    return scorer.run_on_pred(y_true, y_pred, y_proba)
