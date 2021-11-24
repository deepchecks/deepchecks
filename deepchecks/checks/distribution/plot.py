"""A module containing utils for plotting distributions"""
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt


def plot_density(data, xs, color='b', alpha: float = 0.7, **kwargs):
    """Plots a KDE density plot of the data. Adding labels and other plotting attributes is left to ths user

    Args:
        data (): The data used to compute the pdf function.
        xs (iterable): List of x values to plot the computed pdf for.
        color (): Color of the filled graph.
        alpha (float): Transparency of the filled graph.

    Returns:

    """
    density = gaussian_kde(data)
    density.covariance_factor = lambda: .25
    # pylint: disable=protected-access
    density._compute_covariance()
    plt.fill_between(xs, density(xs), color=color, alpha=alpha, **kwargs)
