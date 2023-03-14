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
"""Module for base recsys context."""
import typing as t

import numpy as np
import pandas as pd

from deepchecks.tabular.context import Context as TabularContext
from deepchecks.core.errors import (DatasetValidationError, DeepchecksNotSupportedError, DeepchecksValueError,
                                    ModelValidationError)
from deepchecks.tabular._shared_docs import docstrings
from deepchecks.tabular.dataset import Dataset
from deepchecks.tabular.metric_utils import DeepcheckScorer, get_default_scorers, init_validate_scorers
from deepchecks.tabular.metric_utils.scorers import validate_proba
from deepchecks.tabular.utils.feature_importance import (calculate_feature_importance_or_none,
                                                         validate_feature_importance)
from deepchecks.tabular.utils.task_inference import (get_all_labels, infer_classes_from_model,
                                                     infer_task_type_by_class_number, infer_task_type_by_labels)
from deepchecks.tabular.utils.task_type import TaskType
from deepchecks.tabular.utils.validation import (ensure_predictions_proba, ensure_predictions_shape,
                                                 model_type_validation, validate_model)
from deepchecks.utils.docref import doclink
from deepchecks.utils.logger import get_logger
from deepchecks.utils.plot import DEFAULT_DATASET_NAMES
from deepchecks.utils.typing import BasicModel

__all__ = [
    'Context'
]

@docstrings
class Context(TabularContext):
    """Contains all the data + properties the user has passed to a check/suite, and validates it seamlessly.

    Parameters
    ----------
    train: Union[Dataset, pd.DataFrame, None] , default: None
        Dataset or DataFrame object, representing data an estimator was fitted on
    test: Union[Dataset, pd.DataFrame, None] , default: None
        Dataset or DataFrame object, representing data an estimator predicts on
    model: Optional[BasicModel] , default: None
        A scikit-learn-compatible fitted estimator instance
    {additional_context_params:indent}
    """

    def __init__(
            self,
            train: t.Union[Dataset, pd.DataFrame, None] = None,
            test: t.Union[Dataset, pd.DataFrame, None] = None,
            model: t.Optional[BasicModel] = None,
            feature_importance: t.Optional[pd.Series] = None,
            feature_importance_force_permutation: bool = False,
            feature_importance_timeout: int = 120,
            with_display: bool = True,
            y_pred_train: t.Optional[np.ndarray] = None,
            y_pred_test: t.Optional[np.ndarray] = None,
            y_proba_train: t.Optional[np.ndarray] = None,
            y_proba_test: t.Optional[np.ndarray] = None,
            model_classes: t.Optional[t.List] = None,
    ):

    super().__init__(train, test, model, feature_importance, feature_importance_force_permutation,
                     feature_importance_timeout, with_display, y_pred_train, y_pred_test,
                     y_proba_train, y_proba_test, model_classes)
