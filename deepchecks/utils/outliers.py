import numpy as np

from deepchecks.core.errors import DeepchecksValueError


def one_dimension_iqr(data: np.ndarray, scale: float = 1.5):
    """Calculate outliers on the data given using IQR.

    Returns
    -------
    np.array
        2D numpy array with data given as first column and boolean value of "is outlier" as second column
    """
    data = data.squeeze()
    if data.ndim > 1:
        raise DeepchecksValueError(f'IQR outlier method must receive one dimensional data but got {data.ndim} dims.')
    q1, q3 = np.quantile(data, (.25, .75))
    iqr = q3 - q1
    low_lim = q1 - scale * iqr
    up_lim = q3 + scale * iqr

    outliers = (data < low_lim) | (data > up_lim)
    return outliers
