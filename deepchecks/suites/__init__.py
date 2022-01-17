import logging

logger = logging.getLogger("deepchecks")

logger.warning("DeprecationWarning: Accessing suites directly from deepchecks.suites is deprecated and "
               "will be removed in future releases. Please use deepchecks.tabular.suites for tabular suites.")
from ..tabular.suites import *
