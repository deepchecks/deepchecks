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
from typing import Union

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer

from deepchecks.core.errors import DeepchecksNotImplementedError
from deepchecks.tabular import Dataset
from deepchecks.tabular.utils.task_type import TaskType


def calculate_per_sample_loss(model, task_type: TaskType, dataset: Dataset,
                              classes_index_order: Union[np.array, pd.Series, None] = None) -> pd.Series:
    """Calculate error per sample for a given model and a dataset."""
    if task_type == TaskType.REGRESSION:
        return pd.Series([metrics.mean_squared_error([y], [y_pred]) for y, y_pred in
                          zip(model.predict(dataset.features_columns), dataset.label_col)], index=dataset.data.index)
    else:
        if not classes_index_order:
            if hasattr(model, 'classes_'):
                classes_index_order = model.classes_
            else:
                raise DeepchecksNotImplementedError(
                    'Could not infer classes index order. Please provide them via the classes_index_order '
                    'argument. Alternatively, provide loss_per_sample vector as an argument to the check.')
        proba = model.predict_proba(dataset.features_columns)
        return pd.Series([metrics.log_loss([y], [y_proba], labels=classes_index_order) for
                          y_proba, y in zip(proba, dataset.label_col)], index=dataset.data.index)


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

    if transformed_labels.shape[1] == 1:
        transformed_labels = np.append(
            1 - transformed_labels, transformed_labels, axis=1
        )

    # clip and renormalization y_pred
    y_pred = y_pred.astype('float').clip(eps, 1 - eps)
    y_pred /= y_pred.sum(axis=1)[:, np.newaxis]

    return -(transformed_labels * np.log(y_pred)).sum(axis=1)


def per_sample_mse(y_true, y_pred):
    """Calculate mean square error on a single value."""
    return (y_true - y_pred) ** 2
