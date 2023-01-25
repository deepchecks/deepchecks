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
"""Module for defining the deepchecks CustomMetric abstract class."""
import abc

__all__ = ['CustomMetric']


class CustomMetric(abc.ABC):
    """Abstract class for defining custom metrics.

    The class defines the interface for custom metrics.
    """

    @abc.abstractmethod
    def reset(self):
        """Reset the metric."""
        pass

    @abc.abstractmethod
    def update(self, output):
        """Update the metric with the output of the model."""
        pass

    @abc.abstractmethod
    def compute(self):
        """Compute the metric."""
        pass
