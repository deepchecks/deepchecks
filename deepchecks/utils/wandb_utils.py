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
# pylint: disable=import-outside-toplevel
"""Wandb utils module."""
from typing import Any

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
    try:
        import wandb
    except ImportError as error:
        raise ImportError(
            '"set_wandb_run_state" requires the wandb python package. '
            'To get it, run "pip install wandb".'
        ) from error
    else:
        if dedicated_run is None:
            dedicated_run = wandb.run is None
        if dedicated_run:
            kwargs['project'] = kwargs.get('project', 'deepchecks')
            kwargs['config'] = kwargs.get('config', default_config)
            wandb.init(**kwargs)
            wandb.run._label(repo='Deepchecks')  # pylint: disable=protected-access
        return dedicated_run
