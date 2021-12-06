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
"""Top module for deepchecks library."""
from .base import *
import matplotlib
import matplotlib.pyplot as plt

from .utils.ipython import is_notebook

# Matplotlib has multiple backends. If we are in a context that does not support GUI (For example, during unit tests)
# we can't use a GUI backend. Thus we must use a non-GUI backend.
if not is_notebook():
    matplotlib.use('Agg')
