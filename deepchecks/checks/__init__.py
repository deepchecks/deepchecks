import logging

logger = logging.getLogger("deepchecks")

logger.warning("DeprecationWarning: Accessing checks directly from deepchecks.checks is deprecated and "
               "will be removed in future releases. Please use deepchecks.tabular.checks for tabular checks.")
from ..tabular.checks import *
