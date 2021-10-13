"""Top module for MLChecks library."""
from .base import *
import matplotlib.pyplot as plt
# This is a TEMPORARY solution because currently we use matplotlib, which does not allow us to control the output
# of the graphs, so if the user is in an interactive mode, graphs may be drawed twice. In the near future, we should
# drop matplotlib and start use plotly for our charts.
plt.ioff()

