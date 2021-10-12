"""Top module for MLChecks library."""
from .base import *
from .checks import *
import matplotlib
import matplotlib.pyplot as plt

from .utils import is_notebook
# This is a TEMPORARY solution because currently we use matplotlib, which does not allow us to control the output
# of the graphs, so if the user is in an interactive mode, graphs may be drawed twice. In the near future, we should
# drop matplotlib and start use plotly for our charts.
plt.ioff()

# Matplotlib has multiple backends. If we are in a context that does not support GUI (For example, during unit tests)
# we can't use a GUI backend. Thus we must use a non-GUI backend.
if not is_notebook():
    matplotlib.use('Agg')
