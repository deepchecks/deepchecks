# ----------------------------------------------------------------------------
# Copyright (C) 2021-2023 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""Utils for GPU management."""
import gc

import torch.cuda


def empty_gpu(device):
    """Empty GPU or MPS memory and run garbage collector."""
    gc.collect()
    device = str(device)
    if 'cuda' in device.lower():
        torch.cuda.empty_cache()
    elif 'mps' in device.lower():
        try:
            from torch import mps  # pylint: disable=import-outside-toplevel

            mps.empty_cache()
        except Exception:  # pylint: disable=broad-except
            pass
