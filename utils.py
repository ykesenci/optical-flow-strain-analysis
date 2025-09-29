import numpy as np
import matplotlib.pyplot as plt
import os
import nd2
import re
import time
import matplotlib.image as mpimg
from hs_josephine import hs_optical_flow, _hs_optical_flow
from variables_jojo import grad_domain
from scipy.ndimage import gaussian_filter
from skimage import filters
from scipy import stats

def extract_num(fname):
    match = re.search(r"(\d+)", fname)
    if match:
        return int(match.group(1))
    else:
        return float("inf")

def plot_images_masks(list_images : list, list_masks : list, frame : int = 0, stack : int = 8):
    assert len(list_images) == len(list_masks)
    fig, axes = plt.subplots(2, len(list_images), figsize=(12, 5))
    
    for i, ax in enumerate(axes[0]):
        ax.imshow(list_images[i][frame,stack])
        ax.axis('off')
    
    for i, ax in enumerate(axes[1]):
        ax.imshow(list_masks[i][frame,stack])
        ax.axis('off')
    
    plt.tight_layout()

indices_to_denoise = {
    "Followers_CTL":[0, 1, 2, 7],
    "Followers_KO":[0, 1, 2, 3, 4, 5, 9],
    "Leaders_CTL": [0, 2, 3, 5, 6, 7, 13, 14, 16],
    "Leaders_KO": [1, 3, 4, 5, 9, 12, 21],
}

def process_flow(video, alpha = 0.005, num_iter = 100, num_warp = 1, sigma = 0, denoise = False, ):
    T, _, _, _ = video.shape
    if denoise:
        video = gaussian_filter(video, sigma)
    video = _normalize_video(video)
    u = []
    for t in range(T-1):
        u.append(hs_optical_flow(video[t], video[t+1], alpha = alpha, num_iter=num_iter, num_warp=num_warp))
    u = np.array(u)
    # Rescale u to adapt for microscope resolution
    u[:,2] = (1/0.165) * u[:,2]
    print("Processed flow")
    return u

def strain(u: np.ndarray) -> np.ndarray:
    """
    Compute infinitesimal strain tensor from a displacement field.

    Parameters
    ----------
    u : np.ndarray
        Displacement field of shape (T, d, Z, X, Y).

    Returns
    -------
    e : np.ndarray
        Strain tensor of shape (T, d, d, Z, X, Y).
    """
    T, d = u.shape[:2]
    e = np.zeros((T, d, d) + u.shape[2:], dtype=u.dtype)

    for t in range(T):
        # grads[j][i] = ∂i u_j, with i=0,...,d-1
        grads = [np.gradient(u[t, j], axis=(0, 1, 2)) for j in range(d)]

        for i in range(d):
            for j in range(d):
                e[t, i, j] = 0.5 * (grads[j][i] + grads[i][j])

    return e

def compute_deformation(list_videos, list_masks, alpha=0.01, num_iter=100, num_warp=1, sigma=0.5, indices = indices_to_denoise):
    list_enorm, deformations = [], []

    for index, (video, mask) in enumerate(zip(list_videos, list_masks)):
        try:
            if index in indices:
                denoise = True
            else:
                denoise = False
            # compute optical flow
            u = process_flow(video, alpha=alpha, num_iter=num_iter, num_warp=num_warp, sigma=sigma, denoise = denoise)

            # strain tensor
            e = strain(u)

            # apply mask + median filter
            enorm = mask[0] * filters.median(
                np.mean(np.linalg.norm(e, axis=(1, 2)), axis=0)
            )

            # compute deformation normalized by volume
            volume = np.count_nonzero(mask[0])
            deformation = enorm.sum() / volume if volume > 0 else np.nan

            # store results
            list_enorm.append(enorm)
            deformations.append(deformation)

        except Exception as err:
            print(f"⚠️ Skipping one item due to error: {err}")
            continue

    return list_enorm, deformations


def _normalize_video(video):
    return (video - video.min()) / (video.max() - video.min())

def plot_boxplot_with_pvalue(list1, list2,
                            list1_name='Group 1',
                            list2_name='Group 2',
                            test_type='t-test',
                            title="Leaders",
                             threshold = 0.2,
                             figsize=(8, 6)):
    """
    Plots boxplots for two lists and annotates the p-value from a statistical test.

    Parameters:
    -----------
    list1, list2 : array-like
        Input data lists to compare.
    list1_name, list2_name : str, optional
        Labels for the boxplots (default: 'Group 1', 'Group 2').
    test_type : str, optional
        Statistical test to use: 't-test' (default) or 'mann-whitney'.
    figsize : tuple, optional
        Figure size (width, height) in inches (default: (8, 6)).

    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure object.
    float
        The computed p-value.
    """

    filtered_1 = [element for element in list1 if element > threshold]
    filtered_2 = [element for element in list2 if element > threshold]

    # Compute p-value
    if test_type == 't-test':
        _, p_value = stats.ttest_ind(filtered_1, filtered_2)
    elif test_type == 'mann-whitney':
        _, p_value = stats.mannwhitneyu(filtered_1, filtered_2)
    else:
        raise ValueError("test_type must be 't-test' or 'mann-whitney'")

    # Create boxplot
    plt.figure(figsize=figsize)
    boxplot = plt.boxplot([filtered_1, filtered_2],
                          patch_artist=True,
                          labels=[list1_name, list2_name])

    # Add p-value annotation
    y_max = max(np.max(filtered_1), np.max(filtered_2))
    plt.text(1.5, y_max * 0.95,
             f'p = {p_value:.4f}',
             ha='center', va='top',
             fontsize=12,
             bbox=dict(facecolor='white', alpha=0.8))

    plt.title(f'{title}')
    plt.ylabel('Values')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    return plt.gcf(), p_value  # Return figure and p-value
