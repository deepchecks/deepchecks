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
"""Wandb utils module."""
from typing import Any

try:
    import wandb

    assert hasattr(wandb, '__version__')  # verify package import not local dir
except (ImportError, AssertionError):
    wandb = None

__all__ = ['set_wandb_run_state']


def set_wandb_run_state(dedicated_run: bool, default_config: dict, **kwargs: Any):
    """Set wandb run state.
    Parameters
    ----------
    dedicated_run : bool
        If to initiate and finish a new wandb run.
        If None it will be dedicated if wandb.run is None.
    default_config : dict
        Default config dict.
    kwargs: Keyword arguments to pass to wandb.init - relevent if wandb_init is True.
            Default project name is deepchecks.
            Default config is the check metadata (params, train/test/ name etc.).
    Returns
    -------
    bool
        If deticated run
    """
    assert wandb, 'Missing wandb dependency, please install wandb'
    if dedicated_run is None:
        dedicated_run = wandb.run is None
    if dedicated_run:
        kwargs['project'] = kwargs.get('project', 'deepchecks')
        kwargs['config'] = kwargs.get('config', default_config)
        wandb.init(**kwargs)
    return dedicated_run