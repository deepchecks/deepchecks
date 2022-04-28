# ----------------------------------------------------------------------------
# Copyright (C) 2021-2022 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
import os
import pickle

script_dir = os.path.dirname(__file__)
abs_file_path = os.path.join(script_dir, 'mnist_preds.pkl')

with open(abs_file_path, 'rb') as f:
    mnist_predictions_dict = pickle.load(f)
