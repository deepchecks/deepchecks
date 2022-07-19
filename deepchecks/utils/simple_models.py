# ----------------------------------------------------------------------------
# Copyright (C) 2022 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""Contains simple models used in checks."""
import numpy as np

__all__ = ['PerfectModel', 'RandomModel', 'ClassificationUniformModel', 'RegressionUniformModel']


def create_proba_result(predictions, classes):
    def prediction_to_proba(y_pred):
        proba = np.zeros(len(classes))
        proba[classes.index(y_pred)] = 1
        return proba

    return np.apply_along_axis(prediction_to_proba, axis=1, arr=predictions.reshape(-1, 1))


class PerfectModel:
    """Model used to perfectly predict from given series of labels."""

    def __init__(self):
        self.labels = None

    def fit(self, X, y):  # pylint: disable=unused-argument,invalid-name
        """Fit model."""
        # The X is not used, but it is needed to be matching to sklearn `fit` signature
        self.labels = y

    def predict(self, X):  # pylint: disable=unused-argument,invalid-name
        """Predict on given X."""
        return self.labels.to_numpy()

    def predict_proba(self, X):  # pylint: disable=invalid-name
        """Predict proba for given X."""
        classes = sorted(self.labels.unique().tolist())
        predictions = self.predict(X)
        return create_proba_result(predictions, classes)


class RandomModel:
    """Model used to randomly predict from given series of labels."""

    def __init__(self):
        self.labels = None

    def fit(self, X, y):  # pylint: disable=unused-argument,invalid-name
        """Fit model."""
        # The X is not used, but it is needed to be matching to sklearn `fit` signature
        self.labels = y

    def predict(self, X):  # pylint: disable=invalid-name
        """Predict on given X."""
        return np.random.choice(self.labels, X.shape[0])

    def predict_proba(self, X):  # pylint: disable=invalid-name
        """Predict proba for given X."""
        classes = sorted(self.labels.unique().tolist())
        predictions = self.predict(X)
        return create_proba_result(predictions, classes)


class ClassificationUniformModel:
    """Model that draws predictions uniformly at random from the list of classes in y."""

    def __init__(self):
        self.unique_labels = None

    def fit(self, X, y):  # pylint: disable=unused-argument,invalid-name
        """Fit model."""
        # The X is not used, but it is needed to be matching to sklearn `fit` signature
        self.unique_labels = np.unique(y)

    def predict(self, X):  # pylint: disable=invalid-name
        """Predict on given X."""
        return np.random.choice(self.unique_labels, X.shape[0])

    def predict_proba(self, X):  # pylint: disable=invalid-name
        """Predict proba for given X."""
        classes = sorted(self.unique_labels.tolist())
        predictions = self.predict(X)
        return create_proba_result(predictions, classes)


class RegressionUniformModel:
    """Model that draws predictions uniformly at random from the range of values in y."""

    def __init__(self):
        self.min_value = None
        self.max_value = None

    def fit(self, X, y):  # pylint: disable=unused-argument,invalid-name
        """Fit model."""
        # The X is not used, but it is needed to be matching to sklearn `fit` signature
        self.min_value = y.min()
        self.max_value = y.max()

    def predict(self, X):  # pylint: disable=invalid-name
        """Predict on given X."""
        return np.random.uniform(self.min_value, self.max_value, X.shape[0])
