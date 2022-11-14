# ----------------------------------------------------------------------------
# Copyright (C) 2021-2022 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------
#
"""Module for displaying images and labels."""

import matplotlib.pyplot as plt


def visualize_vision_data(dataset, n_show: int = 6):
    """Display images and labels from the dataset.

    Parameters:
    -----------
    dataset: VisionData
        The dataset to display.
    n_show: int, default: 6
        Number of images to display.
    """
    sampled_ds = dataset.copy(n_show, shuffle=False, random_state=0)
    batch = next(iter(sampled_ds))
    images = sampled_ds.batch_to_images(batch)
    labels = sampled_ds.batch_to_labels(batch)

    _, m_axs = plt.subplots(2, int(n_show / 2.), figsize=(10 * (n_show / 6.), 6))
    m_axs = m_axs.flatten()
    for i in range(len(images)):
        m_axs[i].imshow(images[i])
        m_axs[i].set_title(sampled_ds.label_id_to_name(int(labels[i])))
        m_axs[i].axis('off')
    plt.show()
