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
"""Module with html utility functions."""
import base64

__all__ = ['imagetag']


def imagetag(img: bytes) -> str:
    """Return html image tag with embedded image."""
    png = base64.b64encode(img).decode('ascii')
    return f'<img src="data:image/png;base64,{png}"/>'
