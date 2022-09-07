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
"""Module with common docstrings."""
from deepchecks.utils.decorators import Substitution

_shared_docs = {}

_shared_docs['additional_context_params'] = """
model_name : str , default: ''
    The name of the model
scorers : Optional[Mapping[str, Metric]] , default: None
    dict of scorers names to a Metric
scorers_per_class : Optional[Mapping[str, Metric]] , default: None
    dict of scorers for classification without averaging of the classes.
    See `scikit-learn docs
    <https://scikit-learn.org/stable/modules/model_evaluation.html#from-binary-to-multiclass-and-multilabel>`__.
device : Union[str, torch.device], default: 'cpu'
    processing unit for use
random_state : int
    A seed to set for pseudo-random functions
with_display : bool , default: True
    flag that determines if checks will calculate display (redundant in some checks).
train_predictions : Optional[Dict[int, Union[Sequence[torch.Tensor], torch.Tensor]]] , default None
    Dictionary of the model prediction over the train dataset (keys are the indexes).
test_predictions : Optional[Dict[int, Union[Sequence[torch.Tensor], torch.Tensor]]] , default None
    Dictionary of the model prediction over the test dataset (keys are the indexes).
""".strip('\n')

_shared_docs['property_aggregation_method_argument'] = """
argument for the reduce_output functionality, decides how to aggregate the individual properties drift scores
for a collective score between 0 and 1. Possible values are:
'mean': Mean of all properties scores.
'none': No averaging. Return a dict with a drift score for each property.
'max': Maximum of all the properties drift scores.
""".strip('\n')

docstrings = Substitution(**_shared_docs)
