import numpy as np
import matplotlib.pyplot as plt
import os
import re
from typing import Union, List, Tuple
import tifffile
from scipy import stats
from itertools import combinations

def extract_num(fname: str) -> Union[int, float]:
    """
    Extracts the first integer found in a filename for numerical sorting.

    Args:
        fname (str): Filename string potentially containing numeric sequences.
            Example: "sample_001.tif" → extracts 1, "image25.npy" → extracts 25.

    Returns:
        Union[int, float]: The first integer found in the filename, or infinity if no number is found.
            Return type is int if a number is found, float('inf') otherwise.

    Notes:
        - Uses regex pattern to find the first sequence of digits.
        - Returns float('inf') for filenames without numbers to ensure they sort last.
        - Useful for sorting files like "img1.tif", "img2.tif", ..., "img10.tif" numerically.

    Example:
        >>> extract_num("sample_005.tif")
        5
        >>> extract_num("imageA.tif")
        inf
    """
    match = re.search(r"(\d+)", fname)
    return int(match.group(1)) if match else float("inf")

def load_files(directory: str) -> List[np.ndarray]:
    """
    Loads all supported image files from a directory as NumPy arrays.
    Supports .npy (NumPy binary) and .tif/.tiff (image) formats, with numerical filename sorting.

    Args:
        directory (str): Path to the directory containing image files.
            Example: "/path/to/images/".

    Returns:
        List[np.ndarray]: List of loaded arrays, ordered numerically by filename.
            Each array corresponds to one image file, with shape depending on the image content.
            Returns empty list if no supported files are found.

    Notes:
        - Files are sorted using `extract_num` to ensure numerical order (e.g., "img1", "img2", ..., "img10").
        - Supported formats:
          - .npy: Loaded via `np.load` (preserves original array structure)
          - .tif/.tiff: Loaded via `tifffile.imread` (returns 2D/3D image arrays)
        - Skips unsupported file types with a warning message.
        - Requires `tifffile` package for TIFF support (install with `pip install tifffile`).

    Example:
        >>> arrays = load_files("/data/images/")
        >>> print(f"Loaded {len(arrays)} arrays with shapes: {[arr.shape for arr in arrays]}")

    Raises:
        FileNotFoundError: If the directory does not exist.
        PermissionError: If files cannot be read due to permission issues.
    """
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")

    files = sorted(os.listdir(directory), key=extract_num)
    arrays = []
    for file in files:
        filepath = os.path.join(directory, file)
        try:
            if file.endswith('.npy'):
                arrays.append(np.load(filepath))
            elif file.endswith(('.tif', '.tiff')):
                arrays.append(tifffile.imread(filepath))
            else:
                print(f"Skipping unsupported file: {file}")
        except (OSError, ValueError, RuntimeError, AttributeError) as e:
            print(f"Warning: Failed to load {file}: {str(e)}")
    return arrays, files


def _normalize_video(video: np.ndarray) -> np.ndarray:
    """
    Normalizes pixel intensities of a video/volume to the [0, 1] range.
    Applies min-max normalization independently to each frame/volume.

    Args:
        video (np.ndarray): Input array of shape (T, Z, X, Y) or (Z, X, Y), where:
            - T: Time dimension (optional, for video sequences)
            - Z, X, Y: Spatial dimensions
            Values can be any numeric type (uint8, uint16, float, etc.).

    Returns:
        np.ndarray: Normalized array with same shape as input, with values in [0, 1].
            Preserves original dtype unless conversion is required.

    Notes:
        - Uses formula: (video - min) / (max - min)
        - Handles both single volumes (3D) and video sequences (4D).
        - If input is constant (min == max), returns array of zeros (avoids division by zero).
        - For multi-channel images, normalize each channel independently.
        - Does not modify the input array (creates a new normalized array).

    Example:
        >>> video = np.random.randint(0, 255, (10, 64, 128, 128), dtype=np.uint8)
        >>> normalized = _normalize_video(video)
        >>> print(normalized.min(), normalized.max())  # Should print: 0.0 1.0

    Warning:
        - Normalization is sensitive to outliers. Consider clipping extreme values first.
    """
    if video.min() == video.max():
        return np.zeros_like(video, dtype=np.float32)

    return (video - video.min()) / (video.max() - video.min())

def plot_boxplot_with_pvalue(
    list1: Union[List[float], np.ndarray],
    list2: Union[List[float], np.ndarray],
    list1_name: str = 'Population 1',
    list2_name: str = 'Population 2',
    test_type: str = 't-test',
    figsize: tuple = (8, 6),
    output_dir: str = None,
    filename: str = "boxplot_comparison.pdf"
) -> Tuple[plt.Figure, Tuple[float, float]]:
    """
    Generates boxplots comparing two datasets and annotates the statistical significance (p-value)
    and effect size (Cohen's d). Saves the plot as a PDF file if output_dir is provided.

    Args:
        list1: First dataset to compare.
        list2: Second dataset to compare.
        list1_name: Label for the first dataset. Default: 'Population 1'.
        list2_name: Label for the second dataset. Default: 'Population 2'.
        test_type: Statistical test to perform ('t-test' or 'mann-whitney'). Default: 't-test'.
        figsize: Figure dimensions (width, height) in inches. Default: (8, 6).
        output_dir: Directory to save the PDF file. If None, the plot is not saved. Default: None.
        filename: Name of the output PDF file. Default: "boxplot_comparison.pdf".

    Returns:
        Tuple[plt.Figure, Tuple[float, float]]:
            - Figure: Matplotlib figure object.
            - Tuple: (p_value, cohens_d) computed from the statistical test.

    Notes:
        - Cohen's d is calculated as the standardized mean difference (SMD).
        - For Mann-Whitney U test, Cohen's d is approximated using rank-biserial correlation.
        - P-value and Cohen's d are displayed above the boxplots.
        - If output_dir is provided, the plot is saved as a PDF file.
    """
    # Input validation
    if not isinstance(list1, (list, np.ndarray)) or not isinstance(list2, (list, np.ndarray)):
        raise TypeError("list1 and list2 must be array-like objects")

    # Convert to numpy arrays for calculations
    arr1 = np.asarray(list1)
    arr2 = np.asarray(list2)

    # Compute p-value and Cohen's d
    if test_type == 't-test':
        _, p_value = stats.ttest_ind(arr1, arr2)
        # Cohen's d for t-test
        n1, n2 = len(arr1), len(arr2)
        pooled_std = np.sqrt(((n1 - 1) * np.std(arr1, ddof=1)**2 +
                             (n2 - 1) * np.std(arr2, ddof=1)**2) / (n1 + n2 - 2))
        cohens_d = (np.mean(arr1) - np.mean(arr2)) / pooled_std
    elif test_type == 'mann-whitney':
        _, p_value = stats.mannwhitneyu(arr1, arr2)
        # Cohen's d approximation for Mann-Whitney U
        n1, n2 = len(arr1), len(arr2)
        rank_biserial = 1 - (2 * stats.mannwhitneyu(arr1, arr2, alternative='two-sided').statistic) / (n1 * n2)
        cohens_d = rank_biserial * np.sqrt((n1 + n2) / (n1 * n2))
    else:
        raise ValueError("test_type must be 't-test' or 'mann-whitney'")

    # Create boxplot
    fig = plt.figure(figsize=figsize)
    box = plt.boxplot([arr1, arr2],
                     patch_artist=True,
                     labels=[list1_name, list2_name])

    # Style boxplots
    colors = ['lightblue', 'lightgreen']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    # Add statistical annotation
    y_max = max(np.nanmax(arr1), np.nanmax(arr2))
    stats_text = (f'p = {p_value:.4f}\n'
                  f"Cohen's d = {cohens_d:.3f} "
                  f'({"small" if abs(cohens_d) < 0.5 else "medium" if abs(cohens_d) < 0.8 else "large"})')

    plt.text(1.5, y_max * 0.95, stats_text,
             ha='center', va='top', fontsize=12,
             bbox=dict(facecolor='white', alpha=0.8))

    plt.title(f"Comparison between {list1_name} and {list2_name}",)
    plt.ylabel('Values')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    # Save the figure if output_dir is provided
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
        fig.savefig(output_path, format='pdf', bbox_inches='tight')
        print(f"Plot saved to: {output_path}")

    return fig, (p_value, cohens_d)

def plot_all_boxplot_pairs(data_dict, output_dir="Figures", **kwargs):
    """
    Plots boxplot comparisons for all possible pairs of keys in a dictionary.

    Args:
        data_dict (dict): Dictionary where keys are population names and values are lists of floats.
        output_dir (str): Directory to save the PDF plots. Default: "Figures".
        **kwargs: Additional arguments to pass to `plot_boxplot_with_pvalue`.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get all unique pairs of keys
    key_pairs = combinations(data_dict.keys(), 2)

    for key_i, key_j in key_pairs:
        list_i = data_dict[key_i]
        list_j = data_dict[key_j]

        # Generate a filename based on the keys
        filename = f"{key_i}_vs_{key_j}.pdf"

        # Call the plotting function
        fig, (p_value, cohens_d) = plot_boxplot_with_pvalue(
            list_i, list_j,
            list1_name=key_i,
            list2_name=key_j,
            output_dir=output_dir,
            filename=filename,
            **kwargs
        )

        print(f"Plotted {key_i} vs {key_j}: p = {p_value:.4f}, Cohen's d = {cohens_d:.3f}")
        plt.close(fig)  # Close the figure to free memory
