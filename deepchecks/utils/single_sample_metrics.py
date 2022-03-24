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
"""Common metrics to calculate performance on single samples."""
import numpy as np
from sklearn.preprocessing import LabelBinarizer


def per_sample_cross_entropy(y_true: np.array, y_pred: np.array, eps=1e-15):
    """Calculate cross entropy on a single sample.

    This code is based on the code for sklearn log_loss metric, without the averaging over all samples. Licence below:
    BSD 3-Clause License

    Copyright (c) 2007-2021 The scikit-learn developers.
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this
      list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.

    * Neither the name of the copyright holder nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    """
    y_true = np.array(y_true)

    # Make y_true into one-hot
    # We assume that the integers in y_true correspond to the columns in y_pred, such that if y_true[i] = k, then
    # the corresponding predicted probability would be y_pred[i, k]
    lb = LabelBinarizer()
    lb.fit(list(range(y_pred.shape[1])))
    transformed_labels = lb.transform(y_true)

    # clip and renormalization y_pred
    y_pred = y_pred.clip(eps, 1 - eps)
    y_pred /= y_pred.sum(axis=1)[:, np.newaxis]

    return -(transformed_labels * np.log(y_pred)).sum(axis=1)


def per_sample_mse(y_true, y_pred):
    """Calculate mean square error on a single value."""
    return (y_true - y_pred) ** 2
