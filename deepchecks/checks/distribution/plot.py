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
"""A module containing utils for plotting distributions."""
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt


def plot_density(data, xs, color='b', alpha: float = 0.7, **kwargs) -> np.ndarray:
    """Plot a KDE density plot of the data. Adding labels and other plotting attributes is left to ths user.

    Args:
        data (): The data used to compute the pdf function.
        xs (iterable): List of x values to plot the computed pdf for.
        color (): Color of the filled graph.
        alpha (float): Transparency of the filled graph.

    Returns:
        np.array: The computed pdf values at the points xs.
    """
    density = gaussian_kde(data)
    density.covariance_factor = lambda: .25
    # pylint: disable=protected-access
    density._compute_covariance()
    plt.fill_between(xs, density(xs), color=color, alpha=alpha, **kwargs)
    plt.gca().set_ylim(bottom=0)

    return density(xs)
