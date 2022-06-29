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
import logging

from .tabular_bench import BenchmarkTabular
from .vision_bench import BenchmarkVision

__all__ = ['BenchmarkTabular', 'BenchmarkVision']

logging.getLogger('deepchecks').setLevel(logging.ERROR)
