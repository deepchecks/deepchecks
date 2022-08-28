# -*- coding: utf-8 -*-
"""
.. _plot_tabular_identifier_label_correlation:

Identifier Label Correlation
****************************

This notebook provides an overview for using and understanding the identifier-label correlation check.

This check computes the Predictive Power Score (:ref:`PPS
<plot_tabular_feature_label_correlation_change__how_is_the_predictive_power_score_pps_calculated>`)
meaning, the ability of a unique identifier (index or datetime) column to predict the label.

High predictive score could indicate a problem in the data collection pipeline, and even though the identifier column
doesn't directly enter the model, collecting the data differently for different labels could have an indirect influence
on the data.

**Structure:**

* `Generate Data <#generate-data>`__
* `Run the Check <#run-the-check>`__
* `Define a Condition <#define-a-condition>`__
"""

#%%
# Imports
# =======

import numpy as np
import pandas as pd

from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import IdentifierLabelCorrelation

#%%
# Generate Data
# ===============

np.random.seed(42)
df = pd.DataFrame(np.random.randn(100, 3), columns=['x1', 'x2', 'x3'])
df['x4'] = df['x1'] * 0.05 + df['x2']
df['x5'] = df['x2']*121 + 0.01 * df['x1']
df['label'] = df['x5'].apply(lambda x: 0 if x < 0 else 1)

#%%

dataset = Dataset(df, label='label', index_name='x1', datetime_name='x2', cat_features=[])

#%%
# Run The Check
# ==============

check = IdentifierLabelCorrelation()
check.run(dataset)

# To display the results in an IDE like PyCharm, you can use the following code:
# check.run(ds).show()
# The result will be displayed in a new window.

#%%
# Define a Condition
# ==================
# Now we will define a condition that the PPS should be less than or equal to 0.2.
result = check.add_condition_pps_less_or_equal(max_pps=0.2).run(dataset)
result.show(show_additional_outputs=False)
