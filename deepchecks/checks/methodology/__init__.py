# ----------------------------------------------------------------------------
# Copyright (C) 2021 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""Module contains checks for methodological flaws in the model building process."""
from .performance_overfit import *
from .boosting_overfit import *
from .unused_features import *
from .single_feature_contribution import *
from .single_feature_contribution_train_validation import *
from .index_leakage import *
from .train_test_samples_mix import *
from .date_train_test_leakage_duplicates import *
from .date_train_test_leakage_overlap import *
from .identifier_leakage import *
from .model_inference_time import *
from .datasets_size_comparison import *
