import logging

logger = logging.getLogger('deepchecks')

try:
    import torch
except ImportError:
    logger.error("PyTorch is not installed. Please install it in order to use deepchecks.vision")
