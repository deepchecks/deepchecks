"""Top module for deepchecks library."""
from .base import *
import matplotlib
import matplotlib.pyplot as plt

from .utils.ipython import is_notebook

# Matplotlib has multiple backends. If we are in a context that does not support GUI (For example, during unit tests)
# we can't use a GUI backend. Thus we must use a non-GUI backend.
if not is_notebook():
    matplotlib.use('Agg')
