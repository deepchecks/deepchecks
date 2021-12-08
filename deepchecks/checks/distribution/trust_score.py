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
"""Module of trust score confidence method.

Code is taken from https://github.com/google/TrustScore
Based on: arXiv:1805.11783 [stat.ML]

Used according to the following License:

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
# pylint: disable=invalid-name
import warnings
from typing import Tuple

import numpy as np
from sklearn.neighbors import KDTree, KNeighborsClassifier

__all__ = ['TrustScore']


class TrustScore:
    """Calculate trust score.

    Args:
        k_filter (int): Number of neighbors used during either kNN distance or probability filtering.
        alpha (float): Fraction of instances to filter out to reduce impact of outliers.
        filter_type (str): Filter method; either 'distance_knn' or 'probability_knn'
        leaf_size (int): Number of points at which to switch to brute-force. Affects speed and memory required to
                         build trees. Memory to store the tree scales with n_samples / leaf_size.
        metric (str): Distance metric used for the tree. See sklearn's DistanceMetric class for a list of available
                      metrics.
        dist_filter_type (str): Use either the distance to the k-nearest point (dist_filter_type = 'point') or
                                the average distance from the first to the k-nearest point in the data
                                (dist_filter_type = 'mean').
    """

    def __init__(self, k_filter: int = 10, alpha: float = 0., filter_type: str = 'distance_knn',
                 leaf_size: int = 40, metric: str = 'euclidean', dist_filter_type: str = 'point') -> None:
        super().__init__()
        self.k_filter = k_filter
        self.alpha = alpha
        self.filter = filter_type
        self.eps = 1e-12
        self.leaf_size = leaf_size
        self.metric = metric
        self.dist_filter_type = dist_filter_type

    def filter_by_distance_knn(self, X: np.ndarray) -> np.ndarray:
        """Filter out instances with low kNN density.

        Calculate distance to k-nearest point in the data for each instance and remove instances above a cutoff
        distance.

        Args:
            X (np.ndarray): Data to filter

        Returns:
            (np.ndarray): Filtered data
        """
        kdtree = KDTree(X, leaf_size=self.leaf_size, metric=self.metric)
        knn_r = kdtree.query(X, k=self.k_filter + 1)[0]  # distances from 0 to k-nearest points
        if self.dist_filter_type == 'point':
            knn_r = knn_r[:, -1]
        elif self.dist_filter_type == 'mean':
            knn_r = np.mean(knn_r[:, 1:], axis=1)  # exclude distance of instance to itself
        cutoff_r = np.percentile(knn_r, (1 - self.alpha) * 100)  # cutoff distance
        X_keep = X[np.where(knn_r <= cutoff_r)[0], :]  # define instances to keep
        return X_keep

    def filter_by_probability_knn(self, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Filter out instances with high label disagreement amongst its k nearest neighbors.

        Args:
            X (np.ndarray): Data
            Y (np.ndarray): Predicted class labels

        Returns:
            (np.ndarray, np.ndarray): Filtered data and labels.
        """
        if self.k_filter == 1:
            warnings.warn('Number of nearest neighbors used for probability density filtering should '
                          'be >1, otherwise the prediction probabilities are either 0 or 1 making '
                          'probability filtering useless.')
        # fit kNN classifier and make predictions on X
        clf = KNeighborsClassifier(n_neighbors=self.k_filter, leaf_size=self.leaf_size, metric=self.metric)
        clf.fit(X, Y)
        preds_proba = clf.predict_proba(X)
        # define cutoff and instances to keep
        preds_max = np.max(preds_proba, axis=1)
        cutoff_proba = np.percentile(preds_max, self.alpha * 100)  # cutoff probability
        keep_id = np.where(preds_max >= cutoff_proba)[0]  # define id's of instances to keep
        X_keep, Y_keep = X[keep_id, :], Y[keep_id]
        return X_keep, Y_keep

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Build KDTrees for each prediction class.

        Args:
            X (np.ndarray): Data.
            Y (np.ndarray): Target labels, either one-hot encoded or the actual class label.
        """
        if len(X.shape) > 2:
            warnings.warn(f'Reshaping data from {X.shape} to {X.reshape(X.shape[0], -1).shape} so k-d trees can '
                          'be built.')
            X = X.reshape(X.shape[0], -1)

        # make sure Y represents predicted classes, not one-hot encodings
        if len(Y.shape) > 1:
            Y = np.argmax(Y, axis=1)
            self.classes = Y.shape[1]
        else:
            self.classes = len(np.unique(Y))
        # KDTree and kNeighborsClassifier need 2D data
        self.kdtrees = [None] * self.classes  # type: Any
        self.X_kdtree = [None] * self.classes  # type: Any

        if self.filter == 'probability_knn':
            X_filter, Y_filter = self.filter_by_probability_knn(X, Y)

        for c in range(self.classes):

            if self.filter is None:
                X_fit = X[np.where(Y == c)[0]]
            elif self.filter == 'distance_knn':
                X_fit = self.filter_by_distance_knn(X[np.where(Y == c)[0]])
            elif self.filter == 'probability_knn':
                X_fit = X_filter[np.where(Y_filter == c)[0]]
            else:
                raise Exception('self.filter must be one of ["distance_knn", "probability_knn", None]')

            no_x_fit = len(X_fit) == 0
            if no_x_fit or len(X[np.where(Y == c)[0]]) == 0:
                if no_x_fit and len(X[np.where(Y == c)[0]]) == 0:
                    warnings.warn(f'No instances available for class {c}')
                elif no_x_fit:
                    warnings.warn(f'Filtered all the instances for class {c}. Lower alpha or check data.')
            else:
                self.kdtrees[c] = KDTree(X_fit, leaf_size=self.leaf_size,
                                         metric=self.metric)  # build KDTree for class c
                self.X_kdtree[c] = X_fit

    def score(self, X: np.ndarray, Y: np.ndarray, k: int = 2, dist_type: str = 'point') \
            -> Tuple[np.ndarray, np.ndarray]:
        """Calculate trust scores.

        ratio of distance to closest class other than the predicted class to distance to predicted class.

        Args:
            X (np.ndarray): Instances to calculate trust score for.
            Y (np.ndarray): Either prediction probabilities for each class or the predicted class.
            k (int): Number of nearest neighbors used for distance calculation.
            dist_type (str): Use either the distance to the k-nearest point (dist_type = 'point') or the average
                             distance from the first to the k-nearest point in the data (dist_type = 'mean').

        Returns:
            (np.ndarray, np.ndarray): Batch with trust scores and the closest not predicted class.
        """
        # make sure Y represents predicted classes, not probabilities
        if len(Y.shape) > 1:
            Y = np.argmax(Y, axis=1)

        # KDTree needs 2D data
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)

        d = np.tile(None, (X.shape[0], self.classes))  # init distance matrix: [nb instances, nb classes]

        for c in range(self.classes):
            if (self.kdtrees[c] is None) or (self.kdtrees[c].data.shape[0] < k):
                d[:, c] = np.inf
            else:
                d_tmp = self.kdtrees[c].query(X, k=k)[0]  # get k nearest neighbors for each class
                if dist_type == 'point':
                    d[:, c] = d_tmp[:, -1]
                elif dist_type == 'mean':
                    d[:, c] = np.nanmean(d_tmp[np.isfinite(d_tmp)], axis=1)

        sorted_d = np.sort(d, axis=1)  # sort distance each instance in batch over classes
        # get distance to predicted and closest other class and calculate trust score
        d_to_pred = d[range(d.shape[0]), Y]
        d_to_closest_not_pred = np.where(sorted_d[:, 0] != d_to_pred, sorted_d[:, 0], sorted_d[:, 1])
        trust_score = d_to_closest_not_pred / (d_to_pred + self.eps)
        # closest not predicted class
        class_closest_not_pred = np.where(d == d_to_closest_not_pred.reshape(-1, 1))[1]
        return trust_score, class_closest_not_pred

    @staticmethod
    def process_confidence_scores(baseline_scores: np.ndarray, test_scores: np.ndarray):
        """Process confidence scores."""
        # code to filter extreme confidence values
        filter_center_factor = 4
        filter_center_size = 40.  # % of data

        baseline_confidence = baseline_scores
        if test_scores is None:
            test_confidence = baseline_scores
        else:
            test_confidence = test_scores

        center_size = max(np.nanpercentile(baseline_confidence, 50 + filter_center_size / 2),
                          np.nanpercentile(test_confidence, 50 + filter_center_size / 2)) - \
                      min(np.nanpercentile(baseline_confidence, 50 - filter_center_size / 2),
                          np.nanpercentile(test_confidence, 50 - filter_center_size / 2))
        max_median = max(np.nanmedian(baseline_confidence), np.nanmedian(test_confidence))
        min_median = min(np.nanmedian(baseline_confidence), np.nanmedian(test_confidence))

        upper_thresh = max_median + filter_center_factor * center_size
        lower_thresh = min_median - filter_center_factor * center_size

        baseline_confidence[(baseline_confidence > upper_thresh) | (baseline_confidence < lower_thresh)] = np.nan
        test_confidence[(test_confidence > upper_thresh) | (test_confidence < lower_thresh)] = np.nan

        baseline_confidence = baseline_confidence.astype(float)
        test_confidence = test_confidence.astype(float)

        if test_scores is None:
            test_confidence = None

        return baseline_confidence, test_confidence
