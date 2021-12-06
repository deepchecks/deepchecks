# Deepchecks
# Copyright (C) 2021 Deepchecks
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
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

