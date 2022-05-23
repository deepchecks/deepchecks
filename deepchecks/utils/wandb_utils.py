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
import contextlib
import typing as t

__all__ = ['wandb_run']


@contextlib.contextmanager
def wandb_run(
    project: t.Optional[str] = None,
    use_existing: bool = False,
    **kwargs
) -> t.Iterator[t.Any]:
    """Create new one or use existing wandb run instance.

    Parameters
    ----------
    project : Optional[str], default None
        project name
    use_existing : bool, default False
        whether to create a separate wandb run or use existing one
    **kwargs :
        additional parameters that will be passed to the 'wandb.init'

    Returns
    -------
    Iterator[wandb.sdk.wandb_run.Run]
    """
    try:
        import wandb
    except ImportError as error:
        raise ImportError(
            '"wandb_run" requires the wandb python package. '
            'To get it, run "pip install wandb".'
        ) from error
    else:
        if wandb.run is not None and use_existing is True:
            yield wandb.run
        else:
            kwargs = {'project': project or 'deepchecks', **kwargs}
            with t.cast(t.ContextManager, wandb.init(**kwargs)) as run:
                wandb.run._label(repo='Deepchecks')  # pylint: disable=protected-access
                yield run
