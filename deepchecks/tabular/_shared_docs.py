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
"""Module with common docstrings."""
from deepchecks.utils.decorators import Substitution

_shared_docstrings = {}

_shared_docstrings['additional_context_params'] = """
feature_importance: pd.Series , default: None
    pass manual features importance
feature_importance_force_permutation : bool , default: False
    force calculation of permutation features importance
feature_importance_timeout : int , default: 120
    timeout in second for the permutation features importance calculation
y_pred_train: Optional[np.ndarray] , default: None
    Array of the model prediction over the train dataset.
y_pred_test: Optional[np.ndarray] , default: None
    Array of the model prediction over the test dataset.
y_proba_train: Optional[np.ndarray] , default: None
    Array of the model prediction probabilities over the train dataset.
y_proba_test: Optional[np.ndarray] , default: None
    Array of the model prediction probabilities over the test dataset.
model_classes: Optional[List] , default: None
    For classification: list of classes known to the model
""".strip('\n')

_shared_docstrings['feature_aggregation_method_argument'] = """
Argument for the reduce_output functionality, decides how to aggregate the vector of `per-feature scores` into a
single aggregated score. The aggregated score value is between 0 and 1 for all methods.
Possible values are:
'l3_weighted': Default. L3 norm over the 'per-feature scores' vector weighted by the feature importance, specifically,
sum(FI * PER_FEATURE_SCORES^3)^(1/3). This method takes into account the feature importance yet puts more weight on
the per-feature scores. This method is recommended for most cases.
'l5_weighted': Similar to 'l3_weighted', but with L5 norm. Puts even more emphasis on the per-feature scores and
specifically on the largest per-feature scores returning a score closer to the maximum among the per-feature scores.
'weighted': Weighted mean of per-feature scores based on feature importance.
'max': Maximum of all the per-feature scores.
None: No averaging. Return a dict with a per-feature score for each feature.
""".strip('\n')

docstrings = Substitution(**_shared_docstrings)
