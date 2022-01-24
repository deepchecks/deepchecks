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
"""Utils module containing utilities for plotting."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap


__all__ = ['create_colorbar_barchart_for_check', 'shifted_color_map', 'colors']

colors = {'Train': 'darkblue',
          'Test': '#69b3a2'}


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
