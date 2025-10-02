import numpy as np
from scipy.ndimage import gaussian_filter
from optical_flow import hs_optical_flow
from utils import _normalize_video
from tqdm import tqdm
from skimage.filters import median  # pylint: disable=no-name-in-module

def process_flow(
    video: np.ndarray,
    alpha: float = 0.01,
    num_iter: int = 100,
    num_warp: int = 1,
    sigma: float = 0.5,
    denoise: bool = False,
    z_resolution: float = 1 / 0.165,
    video_index: int = 1,
) -> np.ndarray:
    """
    Computes optical flow between consecutive frames of a video sequence using the Horn-Schunck algorithm.
    Optionally applies Gaussian denoising and normalizes input frames before flow computation.
    Includes resolution scaling for the Z-axis to account for anisotropic voxel sizes in microscopy.

    Args:
        video (np.ndarray): Input video sequence with shape (T, Z, X, Y), where:
            - T: Number of time frames
            - Z, X, Y: Spatial dimensions (3D volume)
        alpha (float, optional): Regularization parameter controlling flow smoothness.
            Lower values allow more flow variation. Default: 0.01.
        num_iter (int, optional): Number of iterations for flow refinement. Default: 100.
        num_warp (int, optional): Number of warping steps for iterative refinement. Default: 1.
        sigma (float, optional): Standard deviation for Gaussian denoising.
            Applied only if denoise=True. Default: 0.5.
        denoise (bool, optional): Whether to apply Gaussian filtering to input video.
            Default: False.
        z_resolution (float, optional): Physical size of a voxel in the Z dimension (in same units as X,Y).
            Used to scale Z-axis flow components to account for anisotropic resolution.
            Default: 1 / 0.165 (typical for many microscopy setups where Z resolution is coarser than XY).

    Returns:
        np.ndarray: Computed flow field with shape (T-1, 3, Z, X, Y), where:
            - T-1: Number of flow fields (one between each consecutive frame pair)
            - 3: Displacement components (Z, X, Y)
            - Z, X, Y: Spatial dimensions matching input
            - Flow in Z-direction is scaled by 1/z_resolution to account for anisotropic voxel sizes

    Notes:
        - Uses `hs_optical_flow` for pairwise frame processing
        - Normalizes video intensities via `_normalize_video` before flow computation
        - Applies Z-axis scaling: flow_z_scaled = flow_z_raw / z_resolution
        - For sigma > 0, applies Gaussian smoothing to each frame when denoise=True
        - Returns one fewer flow fields than input frames (T-1)

    Raises:
        ValueError: If video has fewer than 2 frames (cannot compute flow)
        RuntimeError: If flow computation fails for any frame pair

    Example:
        >>> video = np.random.rand(10, 64, 128, 128)  # 10 frames, 64x128x128 volume
        >>> flow = process_flow(video, z_resolution=0.2)  # Custom Z resolution
    """
    T, _, _, _ = video.shape
    if T < 2:
        raise ValueError("Video must contain at least 2 frames to compute flow")
    
    video = _normalize_video(video)

    if denoise and sigma > 0:
        video = gaussian_filter(video, sigma=sigma)
        video = _normalize_video(video)

    u = []

    for t in tqdm(range(T-1), desc=f"Computing displacement field of video {video_index}"):
        try:
            flow = hs_optical_flow(
                video[t],
                video[t+1],
                alpha=alpha,
                num_iter=num_iter,
                num_warp=num_warp
            )
            u.append(flow)
        except Exception as e:
            raise RuntimeError(f"Flow computation failed between frames {t} and {t+1}: {e}") from e

    u = np.array(u)

    # Apply Z-resolution scaling if Z dimension exists
    if u.shape[1] >= 3:  # Check if Z dimension exists in flow
        u[:, 0] = z_resolution * u[:, 0]  # Scale Z component

    return u

def strain(u: np.ndarray) -> np.ndarray:
    """
    Computes the infinitesimal strain tensor from a displacement field using the symmetric gradient.
    The strain tensor ε is defined as ε = 0.5(∇u + ∇uᵀ), where ∇u is the displacement gradient tensor.
    This captures local deformation including stretching and shearing.

    Args:
        u (np.ndarray): Displacement field with shape (T, d, Z, X, Y), where:
            - T: Time dimension (number of frames).
            - d: Spatial dimensionality (2 or 3 for 2D/3D displacements).
            - Z, X, Y: Spatial dimensions of the displacement field.

    Returns:
        np.ndarray: Strain tensor with shape (T, d, d, Z, X, Y), where:
            - The first two dimensions represent the symmetric d×d strain matrix at each point.
            - The last three dimensions match the spatial dimensions of the input.
            - ε[i,j] = 0.5(∂u_j/∂x_i + ∂u_i/∂x_j) for i,j ∈ {0,...,d-1}.

    Notes:
        - Uses centered finite differences (via `np.gradient`) to compute spatial derivatives.
        - For 2D inputs (d=2), returns [ε_xx, ε_xy; ε_yx, ε_yy].
        - For 3D inputs (d=3), includes shear components (ε_xz, ε_yz, etc.).
        - Assumes uniform spacing between grid points (no physical scaling applied).
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

def compute_deformation(
    list_videos: list[np.ndarray],
    list_masks: list[np.ndarray],
    alpha: float = 0.005,
    num_iter: int = 100,
    num_warp: int = 1,
    sigma: float = 0.5,
    indices: set[int] = None
) -> tuple[list[np.ndarray], list[float]]:
    """
    Computes deformation metrics from video sequences using optical flow and strain analysis.
    For each video-mask pair, calculates:
    1) Optical flow via Horn-Schunck algorithm (with optional denoising)
    2) Strain tensor from the flow field
    3) Energy norm of strain (masked and median-filtered)
    4) Volume-normalized deformation score

    Args:
        list_videos (list[np.ndarray]): List of input videos with shape (T, Z, X, Y).
        list_masks (list[np.ndarray]): List of binary masks with shape (1, Z, X, Y), where:
            - 1 indicates valid regions
            - 0 indicates regions to exclude from analysis.
        alpha (float, optional): Regularization weight for optical flow computation.
            Default: 0.005.
        num_iter (int, optional): Number of iterations for optical flow refinement.
            Default: 100.
        num_warp (int, optional): Number of warping steps in optical flow.
            Default: 1.
        sigma (float, optional): Smoothing parameter for flow preprocessing.
            Default: 0.5.
        indices (set[int], optional): Indices of videos requiring denoising. If None, no denoising.
            Default: None.

    Returns:
        tuple[list[np.ndarray], list[float]]:
            - list_enorm: List of energy norm maps (strain magnitude) with shape (Z, X, Y).
            - deformations: List of volume-normalized deformation scores (scalar per video).
              NaN for failed computations.

    Notes:
        - Skips videos that raise exceptions (prints warning but continues processing).
        - Energy norm is computed as L2-norm of strain tensor, averaged over components,
          then median-filtered and masked.
        - Deformation score = (sum of masked energy norm) / (mask volume).
        - Returns NaN for videos where mask volume = 0 (no valid region).

    Raises:
        ValueError: If list_videos and list_masks have different lengths.
    """
    if len(list_videos) != len(list_masks):
        raise ValueError("list_videos and list_masks must have the same length")

    if indices is None:
        indices = set()

    list_enorm, deformations = [], []
    for index, (video, mask) in enumerate(zip(list_videos, list_masks)):
        try:
            denoise = index in indices
            # Compute optical flow
            u = process_flow(
                video,
                alpha=alpha,
                num_iter=num_iter,
                num_warp=num_warp,
                sigma=sigma,
                denoise=denoise,
                video_index=index+1,
            )
            # Strain tensor
            e = strain(u)
            # Apply mask + median filter
            enorm = mask[0] * median(
                np.mean(np.linalg.norm(e, axis=(1, 2)), axis=0)
            )
            # Compute deformation normalized by volume
            volume = np.count_nonzero(mask[0])
            deformation = enorm.sum() / volume if volume > 0 else np.nan

            list_enorm.append(enorm)
            deformations.append(deformation)
        except (ValueError, IOError) as err:
            print(f"⚠️ Skipping video {index+1} due to error: {err}")
            list_enorm.append(np.nan)
            deformations.append(np.nan)
            continue

    return list_enorm, deformations