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
"""Utils module containing utilities for plotting."""
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap

from deepchecks.utils.strings import format_number_if_not_nan

__all__ = ['create_colorbar_barchart_for_check', 'shifted_color_map',
           'create_confusion_matrix_figure', 'colors', 'hex_to_rgba']


colors = {'Train': '#00008b',  # dark blue
          'Test': '#69b3a2',
          'Baseline': '#b287a3',
          'Generated': '#2191FB'}
# iterable for displaying colors on metrics
metric_colors = ['rgb(102, 197, 204)',
                 'rgb(220, 176, 242)',
                 'rgb(135, 197, 95)',
                 'rgb(158, 185, 243)',
                 'rgb(254, 136, 177)',
                 'rgb(201, 219, 116)',
                 'rgb(139, 224, 164)',
                 'rgb(180, 151, 231)']


def create_colorbar_barchart_for_check(
    x: np.ndarray,
    y: np.ndarray,
    ylabel: str = 'Result',
    xlabel: str = 'Features',
    color_map: str = 'RdYlGn_r',
    start: float = 0,
    stop: float = 1.0,
    tick_steps: float = 0.1,
    color_label: str = 'Color',
    color_shift_midpoint: float = 0.5,
    check_name: str = ''
):
    """Output a colorbar barchart using matplotlib.

    Parameters
    ----------
    x: np.ndarray
        array containing x axis data.
    y: np.ndarray
        array containing y axis data.
    ylabel: str , default: Result
        Name of y axis
    xlabel : str , default: Features
        Name of x axis
    color_map : str , default: RdYlGn_r
        color_map name.
        See https://matplotlib.org/stable/tutorials/colors/colormaps.html for more details
    start : float , default: 0
        start of y axis ticks
    stop : float , default: 1.0
        end of y axis ticks
    tick_steps : float , default: 0.1
        step to y axis ticks
    color_shift_midpoint : float , default: 0.5
        midpoint of color map
    check_name : str , default: ''
        name of the check that called this function

    """
    fig, ax = plt.subplots(figsize=(15, 4))  # pylint: disable=unused-variable

    try:
        my_cmap = plt.cm.get_cmap(color_map + check_name)
    except ValueError:
        my_cmap = plt.cm.get_cmap(color_map)
        my_cmap = shifted_color_map(my_cmap, start=start, midpoint=color_shift_midpoint, stop=stop,
                                    name=color_map + check_name)

    cmap_colors = my_cmap(list(y))
    _ = ax.bar(x, y, color=cmap_colors)  # pylint: disable=unused-variable

    sm = ScalarMappable(cmap=my_cmap, norm=plt.Normalize(start, stop))
    sm.set_array([])

    cbar = plt.colorbar(sm)
    cbar.set_label(color_label, rotation=270, labelpad=25)

    plt.yticks(np.arange(start, stop, tick_steps))
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)


def shifted_color_map(cmap, start=0, midpoint=0.5, stop=1.0, name: str = 'shiftedcmap', transparent_from: float = None):
    """Offset the "center" of a colormap.

     Parameters
    ----------
    cmap
        The matplotlib colormap to be altered
    start , default: 0
        Offset from lowest point in the colormap's range.
        Should be between0.0 and 1.0.
    midpoint , default: 0.5
        The new center of the colormap. Defaults to
        0.5 (no shift). Should be between 0.0 and 1.0. In
        general, this should be  1 - vmax/(vmax + abs(vmin))
        For example if your data range from -15.0 to +5.0 and
        you want the center of the colormap at 0.0, `midpoint`
        should be set to  1 - 5/(5 + 15)) or 0.75
    stop , default: 1.0
        Offset from highest point in the colormap's range.
        Should be between0.0 and 1.0.
    name: str , default: shiftedcmap
    transparent_from : float , default: None
        The point between start and stop where the colors will start being transparent.

    """
    if transparent_from is None:
        transparent_from = stop

    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        if transparent_from / midpoint < si:
            cdict['alpha'].append((si, 0.3, 0.3))
        else:
            cdict['alpha'].append((si, a, a))

    newcmap = LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap


def hex_to_rgba(h, alpha):
    """Convert color value in hex format to rgba format with alpha transparency."""
    return 'rgba' + str(tuple([int(h.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)] + [alpha]))


def create_confusion_matrix_figure(confusion_matrix: np.ndarray, x: np.ndarray,
                                   y: np.ndarray, normalized: bool):
    """Create a confusion matrix figure.

    Parameters
    ----------
    confusion_matrix: np.ndarray
        2D array containing the confusion matrix.
    x: np.ndarray
        array containing x axis data.
    y: np.ndarray
        array containing y axis data.
    normalized: bool
        if True will also show normalized values by the true values.

    Returns
    -------
    plotly Figure object
        confusion matrix figure

    """
    if normalized:
        confusion_matrix_norm = confusion_matrix.astype('float') / \
            confusion_matrix.sum(axis=1)[:, np.newaxis] * 100
        z = np.vectorize(format_number_if_not_nan)(confusion_matrix_norm)
        texttemplate = '%{z}%<br>(%{text})'
        colorbar_title = '% out of<br>True Values'
        plot_title = 'Percent Out of True Values (Count)'
    else:
        z = confusion_matrix
        colorbar_title = None
        texttemplate = '%{text}'
        plot_title = 'Value Count'

    fig = go.Figure(data=go.Heatmap(
                x=x,
                y=y,
                z=z,
                text=confusion_matrix,
                texttemplate=texttemplate))
    fig.data[0].colorbar.title = colorbar_title
    fig.update_layout(title=plot_title)
    fig.update_layout(height=600)
    fig.update_xaxes(title='Predicted Value', type='category', scaleanchor='y', constrain='domain')
    fig.update_yaxes(title='True value', type='category', constrain='domain', autorange='reversed')

    return fig
