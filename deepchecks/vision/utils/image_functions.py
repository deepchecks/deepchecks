import numpy as np
import plotly.graph_objects as go

from deepchecks.core.errors import DeepchecksValueError


def numpy_to_image_figure(data: np.ndarray):
    # Image knows to plot only RGB images
    dimension = data.shape[2]
    if dimension == 3:
        return go.Image(z=data, hoverinfo='skip')
    elif dimension == 1:
        return go.Heatmap(z=data.squeeze(), colorscale='gray', hoverinfo='skip')
    else:
        raise DeepchecksValueError(f'Don\'t know to plot images with {dimension} dimensions')


def apply_heatmap_image_properties(fig):
    # In case of heatmap and grayscale images, need to add those properties which on Image exists automatically
    fig.update_yaxes(autorange='reversed', scaleanchor='x', constrain='domain')
    fig.update_xaxes(constrain='domain')
    fig.update_traces(showscale=False)
