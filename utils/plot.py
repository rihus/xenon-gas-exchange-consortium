"""Plotting functions for the project."""

import sys
from typing import Dict, List, Optional, Union, Tuple

import skimage

import numpy as np
from pathlib import Path
# import scipy.io as sio
import matplotlib

import cv2
matplotlib.use("Agg")
from matplotlib import pyplot as plt

# matplotlib.use("TkAgg")
from matplotlib.axes import Axes

from spect import nmr_timefit
from utils import constants, io_utils

sys.path.append("..")

def _to_rgb(c):
    """
    _to_rgb(c)
    Convert an RGB-like triplet to float RGB in [0,1].
    - Accepts [r,g,b] in 0–1 or 0–255.
    - Returns tuple(float, float, float).
    """

    c = np.array(c, dtype=float)
    if c.max() > 1.0:
        c = c / 255.0
    return tuple(c.tolist())

def _colors_for_bins(centers: np.ndarray,
                     thresholds: Optional[List[float]],
                     xlim: float,
                     cmap_dict: Optional[Dict[int, List[float]]],
                     default_color: Tuple[float,float,float]) -> List[Tuple[float,float,float]]:

    """
    _colors_for_bins(centers, thresholds, xlim, cmap_dict, default_color)
    Assign a color per histogram bin based on which threshold segment its center falls in.
    - Segments: [0, t1), [t1, t2), …, [tN, xlim].
    - Uses `cmap_dict` keys 1..N (key 0 is ignored/background).
    - If colors < segments, last color is reused.
    - Returns list of RGB tuples (len == centers.size).
    """

    if thresholds is None or cmap_dict is None:
        return [default_color] * centers.size
    t = np.sort(np.asarray(thresholds, dtype=float))
    t = t[(t >= 0) & (t <= xlim)]
    bounds = np.r_[0.0, t, xlim]
    seg_idx = np.searchsorted(bounds, centers, side='right') - 1
    keys = sorted([k for k in cmap_dict.keys() if k != 0])  # skip bin 0 (background)
    if not keys:
        return [default_color] * centers.size
    cmap_list = [_to_rgb(cmap_dict[k]) for k in keys]
    colors = [cmap_list[min(i, len(cmap_list)-1)] for i in seg_idx]
    return colors

def _load_profile(profile_path: Union[str, Path]):
    """
    Load a compact histogram profile.
    Supports:
      - .mat: expects variables 'edges' (B+1,), 'probs' (B,), optional 'xlim'
      - .npz: expects arrays 'edges' and 'probs' (and optional 'xlim');
              OR a single array 'arr' of shape (2,B) = [centers; probs]
      - .npy: expects array of shape (2,B) = [centers; probs]
    Returns (x, y) where x are bin centers (not edges) and y are probabilities.
    """
    p = Path(profile_path)
    suf = p.suffix.lower()
    if suf == ".mat":
        try:
            import scipy.io as sio
        except ImportError as e:
            raise ImportError("scipy is required to read .mat files (pip install scipy)") from e
        z = sio.loadmat(p)
        if "edges" in z and "probs" in z:
            edges = np.asarray(z["edges"]).ravel()
            probs = np.asarray(z["probs"]).ravel()
            x = 0.5 * (edges[:-1] + edges[1:])
            y = probs
            return x, y
        raise ValueError(".mat profile must contain 'edges' and 'probs'")
    elif suf == ".npz":
        z = np.load(p)
        if "edges" in z and "probs" in z:
            edges = np.asarray(z["edges"]).ravel()
            probs = np.asarray(z["probs"]).ravel()
            x = 0.5 * (edges[:-1] + edges[1:])
            y = probs
            return x, y
        elif "arr" in z:
            arr = np.asarray(z["arr"])
            if arr.ndim == 2 and arr.shape[0] == 2:
                return arr[0], arr[1]
        else:
            # try first array in the npz
            key = list(z.keys())[0]
            arr = np.asarray(z[key])
            if arr.ndim == 2 and arr.shape[0] == 2:
                return arr[0], arr[1]
        raise ValueError(".npz profile must have ('edges','probs') or a 2xB array")
    elif suf == ".npy":
        arr = np.load(p)
        if arr.ndim == 2 and arr.shape[0] == 2:
            return arr[0], arr[1]
        raise ValueError(".npy profile must be 2xB = [centers; probs]")
    else:
        raise ValueError(f"Unsupported profile type: {suf}")


def _merge_rgb_and_gray(gray_slice: np.ndarray, rgb_slice: np.ndarray) -> np.ndarray:
    """Combine the gray scale image with the RGB binning via HSV.

    Args:
        gray_slice (np.ndarray): 2D image slice of grayscale image.
        rgb_slice (_type_): 3D image slice of the RGB grayscale image of shape
            (H, W, C)

    Returns:
        (np.ndarray): merged image slice
    """
    # construct RGB version of gray-level ute
    gray_slice_color = np.dstack((gray_slice, gray_slice, gray_slice))
    # Convert the input image and color mask to HSV
    gray_slice_hsv = skimage.color.rgb2hsv(gray_slice_color)
    rgb_slice_hsv = skimage.color.rgb2hsv(rgb_slice)
    # Replace the hue and saturation of the original image
    # with that of the color mask
    gray_slice_hsv[..., 0] = rgb_slice_hsv[..., 0]
    gray_slice_hsv[..., 1] = rgb_slice_hsv[..., 1]
    mask = (
        (rgb_slice[:, :, 0] == 0)
        & (rgb_slice[:, :, 1] == 0)
        & (rgb_slice[:, :, 2] == 0)
    )
    mask = ~mask
    gray_slice_hsv[mask, :] = rgb_slice_hsv[mask, :]
    colormap = skimage.color.hsv2rgb(gray_slice_hsv)
    return colormap


def map_grey_to_rgb(image: np.ndarray, cmap: Dict[int, np.ndarray]) -> np.ndarray:
    """Map a greyscale image to a RGB image using a colormap.

    Args:
        image (np.ndarray): greyscale image of shape (x, y, z)
        cmap (Dict[int, np.ndarray]): colormap mapping integers to RGB values.
    Returns:
        RGB image of shape (x, y, z, 3)
    """
    rgb_image = np.zeros((image.shape[0], image.shape[1], image.shape[2], 3))
    for key in cmap.keys():
        rgb_image[image == key] = cmap[key]
    return rgb_image


def get_biggest_island_indices(arr: np.ndarray) -> Tuple[int, int]:
    """Get the start and stop indices of the biggest island in the array.

    Args:
        arr (np.ndarray): binary array of 0s and 1s.
    Returns:
        Tuple of start and stop indices of the biggest island.
    """
    # intitialize count
    cur_count = 0
    cur_start = 0

    max_count = 0
    pre_state = 0

    index_start = 0
    index_end = 0

    for i in range(0, np.size(arr)):
        if arr[i] == 0:
            cur_count = 0
            if (pre_state == 1) & (cur_start == index_start):
                index_end = i - 1
            pre_state = 0

        else:
            if pre_state == 0:
                cur_start = i
                pre_state = 1
            cur_count += 1
            if cur_count > max_count:
                max_count = cur_count
                index_start = cur_start

    return index_start, index_end


def map_and_overlay_to_rgb(
    image: np.ndarray, image_background: np.ndarray, cmap: Dict[int, np.ndarray]
) -> np.ndarray:
    """Map a greyscale image to a RGB image using a colormap and combine w/ background.

    Args:
        image (np.ndarray): greyscale image of shape (x, y, z)
        image_background (np.ndarray): greyscale image of shape (x, y, z)
        cmap (Dict[int, np.ndarray]): colormap mapping integers to RGB values.
    Returns:
        RGB image of shape (x, y, z, 3)
    """
    image_rgb = map_grey_to_rgb(image, cmap)
    image_out = np.zeros((image.shape[0], image.shape[1], image.shape[2], 3))
    for i in range(0, image.shape[2]):
        image_out[:, :, i, :] = _merge_rgb_and_gray(
            image_background[:, :, i], image_rgb[:, :, i, :]
        )
    return image_out


def overlay_mask_on_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Overlay the border of a binary mask on a greyscale image in red.

    Args:
        image (np.ndarray): Greyscale image of shape (x, y, z)
        mask (np.ndarray): Binary mask of shape (x, y, z)

    Returns:
        np.ndarray: Overlaid image of shape (x, y, z, 3)
    """
    # divide by the maximum value to normalize to [0, 1]
    image = image / np.max(image)

    def border_mask(mask: np.ndarray) -> np.ndarray:
        mask_dilated = np.zeros_like(mask)
        for i in range(mask.shape[2]):
            mask_dilated[:, :, i] = cv2.dilate(
                mask[:, :, i].astype(np.uint8), np.ones((3, 3)), iterations=1
            )
        return mask_dilated - mask

    border = border_mask(mask)

    image_out = np.zeros((image.shape[0], image.shape[1], image.shape[2], 3))
    for i in range(image.shape[2]):
        image_slice = np.repeat(image[:, :, i][:, :, np.newaxis], 3, axis=2)
        border_slice = border[:, :, i][:, :, np.newaxis]
        image_slice[border_slice[..., 0] == 1] = [1, 0, 0]
        image_out[:, :, i, :] = image_slice

    return image_out


def get_plot_indices(image: np.ndarray, n_slices: int = 16) -> Tuple[int, int]:
    """Get the indices to plot the image.

    Args:
        image (np.ndarray): binary image.
        n_slices (int, optional): number of slices to plot. Defaults to 16.
    Returns:
        Tuple of start and interval indices.
    """
    sum_line = np.sum(np.sum(image, axis=0), axis=0)
    index_start, index_end = get_biggest_island_indices(sum_line > 300)
    flt_inter = (index_end - index_start) // n_slices

    # threshold to decide interval number
    if np.modf(flt_inter)[0] > 0.4:
        index_skip = np.ceil(flt_inter).astype(int)
    else:
        index_skip = np.floor(flt_inter).astype(int)

    return index_start, index_skip


def make_montage(image: np.ndarray, n_slices: int = 16) -> np.ndarray:
    """Make montage of the image.

    Makes montage of the image.
    Assumes the image is of shape (x, y, z, 3).

    Args:
        image (np.ndarray): image to make montage of.
        n_slices (int, optional): number of slices to plot. Defaults to 16.
    Returns:
        Montaged image array.
    """
    # get the shape of the image
    x, y, z, _ = image.shape
    # get the number of rows and columns
    n_rows = 1 if n_slices < 8 else 2
    n_cols = np.ceil(n_slices / n_rows).astype(int)
    # get the shape of the slices
    slice_shape = (x, y)
    # make the montage array
    montage = np.zeros((n_rows * slice_shape[0], n_cols * slice_shape[1], 3))
    # iterate over the slices
    for slice in range(n_slices):
        # get the row and column
        row = slice // n_cols
        col = slice % n_cols
        # get the slice
        slice = image[:, :, slice, :]
        # add to the montage
        montage[
            row * slice_shape[0] : (row + 1) * slice_shape[0],
            col * slice_shape[1] : (col + 1) * slice_shape[1],
            :,
        ] = slice
    return montage


def plot_montage_grey(
    image: np.ndarray, path: str, index_start: int, index_skip: int = 1
):
    """Plot a montage of the image in grey scale.

    Will make a montage of 2x8 of the image in grey scale and save it to the path.
    Assumes the image is of shape (x, y, z) where there are at least 16 slices.
    Otherwise, will plot all slices.

    Args:
        image (np.ndarray): gray scale image to plot of shape (x, y, z)
        path (str): path to save the image.
        index_start (int): index to start plotting from.
        index_skip (int, optional): indices to skip. Defaults to 1.
    """
    # divide by the maximum value
    image = image / np.max(image)
    # stack the image to make it 4D (x, y, z, 3)
    image = np.stack((image, image, image), axis=-1)
    # plot the montage
    index_end = index_start + index_skip * 16
    montage = make_montage(
        image[:, :, index_start:index_end:index_skip, :], n_slices=16
    )
    plt.figure()
    plt.imshow(montage, cmap="gray")
    plt.axis("off")
    plt.savefig(path, transparent=True, bbox_inches="tight", pad_inches=-0.05, dpi=300)
    plt.clf()
    plt.close()


def plot_montage_color(
    image: np.ndarray,
    path: str,
    index_start: int,
    index_skip: int = 1,
    n_slices: int = 16,
):
    """Plot a montage of the image in RGB.

    Will make a montage of default (2x8) of the image in RGB and save it to the path.
    Assumes the image is of shape (x, y, z) where there are at least n_slices.
    Otherwise, will plot all slices.

    Args:
        image (np.ndarray): RGB image to plot of shape (x, y, z, 3).
        path (str): path to save the image.
        index_start (int): index to start plotting from.
        index_skip (int, optional): indices to skip. Defaults to 1.
        n_slices (int, optional): number of slices to plot. Defaults to 16.
    """
    # plot the montage
    index_end = index_start + index_skip * n_slices
    montage = make_montage(
        image[:, :, index_start:index_end:index_skip, :], n_slices=n_slices
    )
    plt.figure()
    plt.imshow(montage, cmap="gray")
    plt.axis("off")
    plt.savefig(path, transparent=True, bbox_inches="tight", pad_inches=-0.05, dpi=300)
    plt.clf()
    plt.close()


def plot_histogram_rbc_osc(
    data: np.ndarray,
    path: str,
    fig_size: Tuple[int, int] = (9, 6),
    xlim: Tuple[float, float] = (-10, 20),
    ylim: Tuple[float, float] = (0, 0.1),
    xticks: List[float] = [-5, 0, 5, 10, 15],
    yticks: List[float] = [0, 0.05, 0.1],
    plot_ref: bool = True,
):
    """Plot histogram of RBC oscillation.

    Args:
        data (np.ndarray): data to plot histogram of.
        path (str): path to save the image.
    """
    fig, ax = plt.subplots(figsize=fig_size)
    data = data.flatten()
    weights = np.ones_like(data) / float(len(data))
    # plot histogram
    _, bins, _ = ax.hist(
        data, bins=50, color=(0, 0.8, 0.8), weights=weights, edgecolor="black"
    )
    ax.set_ylabel("Fraction of Voxels", fontsize=35)
    # define and plot healthy reference line
    if plot_ref:
        data_ref = io_utils.import_np(path="data/reference_dist.npy")
        n, bins, _ = ax.hist(
            data_ref,
            bins=bins,
            color=(1, 1, 1),
            alpha=0.0,
            weights=np.ones_like(data_ref) / float(len(data_ref)),
        )
        ax.plot(1.2 * (bins[1:] + bins[:-1]), n, "--", color="k", linewidth=4)
    # set plot parameters
    plt.xlim(xlim)
    plt.ylim(ylim)
    # define ticks
    plt.xticks(xticks, ["{:.0f}".format(x) for x in xticks], fontsize=40)
    plt.yticks(yticks, ["{:.2f}".format(x) for x in yticks], fontsize=40)
    fig.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_histogram(
    data: np.ndarray,
    path: str,
    color: Tuple[float, float, float],
    xlim: float,
    ylim: float,
    num_bins: int,
    refer_fit: Union[Tuple[float, float, float], str, None] = None,  # healthy ref (optional)
    xticks: Optional[List[float]] = None,
    yticks: Optional[List[float]] = None,
    xticklabels: Optional[List[str]] = None,
    yticklabels: Optional[List[str]] = None,
    xlabel: Optional[str] = None,
    title: Optional[str] = None,
    thresholds: Optional[List[float]] = None,
    thresh_style: dict = None,
    band_colors: Optional[Dict[int, List[float]]] = None,  # per-segment bar colors
    outline: str = "data",                                  # "data" or "none"
    outline_style: Optional[dict] = None,                   # solid outline style
    healthy_style: Optional[dict] = None,                   # dashed healthy-ref style
):
    """
    Plot a publication-style histogram with:
    - Bars colored by threshold segments (via `band_colors` CMAP; bin 0 ignored).
    - Optional solid outline of THIS data’s histogram.
    - Optional dashed “healthy” overlay from a Gaussian (A, μ, σ) or a saved profile (.mat/.npz/.npy).

    Args:
      data (ndarray): 1D values; clipped to [0, xlim].
      path (str): Output image path.
      color (tuple): Base RGB (used for outline/fallback).
      xlim, ylim (float): Axis limits (x in data units; y in probability).
      num_bins (int): Number of bins in [0, xlim].
      refer_fit ((A, μ, σ) | str | None): Gaussian tuple or profile filepath; None = no overlay.
      xticks/yticks (list[float] | None), xticklabels/yticklabels (list[str] | None): Tick spec.
      xlabel/title (str | None): Labels.
      thresholds (list[float] | None): Segment cut points (same units as data/xlim).
      thresh_style (dict | None): Style for vertical threshold lines.
      band_colors (dict[int, list[float]] | None): Segment colors; keys 1..N (0 is background).
      outline ("data" | "none"): Solid outline of data histogram (default "data").
      outline_style/healthy_style (dict | None): Style overrides.

    Notes:
    - Bars are probability-normalized (sum ≈ 1).
    - For RBC/Mem, keep data in raw units; show ×100 only in tick labels if desired.
    """

    plt.rc("axes", linewidth=4)
    fig, ax = plt.subplots(figsize=(9, 6))

    # ----- data prep -----
    d = np.asarray(data, dtype=float).ravel()
    d = np.clip(d, 0.0, xlim)
    d = np.append(d, xlim)  # ensure last bin has ≥1 sample

    # ----- explicit histogram -----
    counts, edges = np.histogram(d, bins=num_bins, range=(0.0, xlim))
    probs   = counts.astype(float) / float(d.size)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths  = np.diff(edges)

    # colored bars
    bar_colors = _colors_for_bins(centers, thresholds, xlim, band_colors, default_color=_to_rgb(color))
    ax.bar(centers, probs, width=widths, align="center",
           color=bar_colors, edgecolor="black", linewidth=1.0, zorder=2)

    # solid outline of THIS histogram
    if outline and outline.lower() == "data":
        st = {"linestyle": "-", "linewidth": 3.0, "color": _to_rgb(color)}
        if outline_style:
            st.update(outline_style)
        ax.step(edges, np.r_[probs, 0.0], where="post", zorder=6, **st)

    # dashed healthy reference overlay (optional)
    if refer_fit is not None:
        ref_st = {"linestyle": "--", "linewidth": 3.0, "color": "k"}
        if healthy_style:
            ref_st.update(healthy_style)
        if isinstance(refer_fit, (str, Path)):
            x_ref, y_ref = _load_profile(refer_fit)
            ax.plot(x_ref, y_ref, zorder=6, **ref_st)
        else:
            A, mu, sigma = refer_fit
            x_ref = edges  # line up visually with our bins
            y_ref = A * np.exp(-(((x_ref - mu) / sigma) ** 2))
            ax.plot(x_ref, y_ref, zorder=6, **ref_st)

    # dashed threshold lines
    if thresholds is not None:
        style = {"color": "k", "linestyle": "--", "linewidth": 2}
        if thresh_style:
            style.update(thresh_style)
        for t in thresholds:
            if 0 <= t <= xlim:
                ax.axvline(t, **style, zorder=7)

    # axes styling
    ax.set_xlim(0, xlim)
    ax.set_ylim(0, ylim)
    plt.locator_params(axis="x", nbins=4)
    try:
        plt.xticks(xticks, xticklabels, fontsize=35)
        plt.yticks(yticks, yticklabels, fontsize=35)
    except TypeError:
        plt.xticks(fontsize=40)
        plt.yticks(fontsize=40)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=30)
    if title is not None:
        ax.set_title(title, fontsize=30)

    fig.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


# def plot_histogram(
#     data: np.ndarray,
#     path: str,
#     color: Tuple[float, float, float],
#     xlim: float,
#     ylim: float,
#     num_bins: int,
#     refer_fit: Tuple[float, float, float],
#     xticks: Optional[List[float]] = None,
#     yticks: Optional[List[float]] = None,
#     xticklabels: Optional[List[str]] = None,
#     yticklabels: Optional[List[str]] = None,
#     xlabel: Optional[str] = None,
#     title: Optional[str] = None,
# ):
#     """Plot histogram of arbitrary data.

#     Args:
#         data (np.ndarray): data to plot histogram of.
#         path (str): path to save the image.
#         color (Tuple[float, float, float]): color of the histogram.
#         xlim (float): x limit of the histogram.
#         ylim (float): y limit of the histogram.
#         num_bins (int): number of bins in the histogram.
#         refer_fit (Tuple[float, float, float]): fit parameters of the healthy reference.
#         xticks (Optional[List[float]], optional): x ticks. Defaults to None.
#         yticks (Optional[List[float]], optional): y ticks. Defaults to None.
#         xticklabels (Optional[List[str]], optional): x tick labels. Defaults to None.
#         yticklabels (Optional[List[str]], optional): y tick labels. Defaults to None.
#         xlabel (Optional[str], optional): x label. Defaults to None.
#     """
#     # make a thick frame
#     plt.rc("axes", linewidth=4)
#     fig, ax = plt.subplots(figsize=(9, 6))
#     # the histogram of the data
#     # limit the range of data
#     data = data.flatten()
#     data[data < 0] = 0
#     data[data > xlim] = xlim
#     data = np.append(data, xlim)
#     weights = np.ones_like(data) / float(len(data))
#     # plot histogram
#     _, bins, _ = ax.hist(
#         data, num_bins, color=color, weights=weights, edgecolor="black"
#     )
#     # define and plot healthy reference line
#     normal = refer_fit[0] * np.exp(-(((bins - refer_fit[1]) / refer_fit[2]) ** 2))
#     ax.plot(bins, normal, "--", color="k", linewidth=4)
#     plt.xlim((0, xlim))
#     plt.ylim((0, ylim))
#     plt.locator_params(axis="x", nbins=4)
#     try:
#         plt.xticks(xticks, xticklabels, fontsize=35)
#         plt.yticks(yticks, yticklabels, fontsize=35)
#     except TypeError:
#         plt.xticks(fontsize=40)
#         plt.yticks(fontsize=40)
#     if xlabel is not None:
#         ax.set_xlabel(xlabel, fontsize=30)
#     if title is not None:
#         ax.set_title(title, fontsize=30)
#     # Tweak spacing to prevent clipping of ylabel
#     fig.tight_layout()
#     plt.savefig(path)
#     plt.close()


def plot_data_rbc_k0(
    t: np.ndarray,
    data: np.ndarray,
    path: str,
    high: np.ndarray = np.array([]),
    low: np.ndarray = np.array([]),
):
    """Plot RBC k0 and binned indices."""
    fig, ax = plt.subplots(figsize=(9, 6))
    # plot healthy reference line
    ax.plot(t, data, "-", color="k", linewidth=5)
    ax.plot(t[high], data[high], ".", color="C2", markersize=10)
    ax.plot(t[low], data[low], ".", color="C1", markersize=10)
    ax.plot(t, np.zeros((len(t), 1)), ".", color="k", linewidth=2)
    ax.set_ylabel("Intensity (au)", fontsize=35)
    # set plot parameters
    plt.rc("axes", linewidth=4)
    plt.xticks([], [])
    plt.yticks(fontsize=40)
    # set ticks
    fig.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_histogram_with_thresholds(
    data: np.ndarray, thresholds: List[float], path: str
):
    """Generate the histogram for the healthy reference distribution.

    Plot histogram of the data with the thresholds each having a different color by
    setting the face color in matplotlib. All values below the first threshold are
    red, all values between the first and second threshold are orange, all values above
    last threshold are purple.

    Args:
        data (np.ndarray): data to plot
        thresholds (List[float]): list of thresholds to plot of length 7.
        path (str): path to save the figure.
    """
    _, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ax.hist(data, bins=500, density=True)

    # Plot the thresholds
    for threshold in thresholds:
        ax.axvline(threshold, color="k", linestyle="--", linewidth=1)
    # Set the face color for the thresholds
    i = 0
    while ax.patches[i].get_x() < thresholds[0]:
        ax.patches[i].set_facecolor((1, 0, 0))  # red
        i += 1

    while (
        i < len(ax.patches)
        and ax.patches[i].get_x() >= thresholds[0]
        and ax.patches[i].get_x() < thresholds[1]
    ):
        ax.patches[i].set_facecolor((1, 0.7143, 0))
        i += 1
    while (
        i < len(ax.patches)
        and ax.patches[i].get_x() >= thresholds[1]
        and ax.patches[i].get_x() < thresholds[2]
    ):
        ax.patches[i].set_facecolor((0.4, 0.7, 0.4))
        i += 1
    while (
        i < len(ax.patches)
        and ax.patches[i].get_x() >= thresholds[2]
        and ax.patches[i].get_x() < thresholds[3]
    ):
        ax.patches[i].set_facecolor((0, 1, 0))
        i += 1
    while (
        i < len(ax.patches)
        and ax.patches[i].get_x() >= thresholds[3]
        and ax.patches[i].get_x() < thresholds[4]
    ):
        ax.patches[i].set_facecolor((184.0 / 255.0, 226.0 / 255.0, 145.0 / 255.0))
        i += 1
    while (
        i < len(ax.patches)
        and ax.patches[i].get_x() >= thresholds[4]
        and ax.patches[i].get_x() < thresholds[5]
    ):
        ax.patches[i].set_facecolor((243.0 / 255.0, 205.0 / 255.0, 213.0 / 255.0))
        i += 1
    while (
        i < len(ax.patches)
        and ax.patches[i].get_x() >= thresholds[5]
        and ax.patches[i].get_x() < thresholds[6]
    ):
        ax.patches[i].set_facecolor((225.0 / 255.0, 129.0 / 255.0, 162.0 / 255.0))
        i += 1
    while i < len(ax.patches) and ax.patches[i].get_x() >= thresholds[6]:
        ax.patches[i].set_facecolor((197.0 / 255.0, 27.0 / 255.0, 125.0 / 255.0))
        i += 1
    # increase the size of the tick labels
    ax.set_xlabel("RBC Oscillation Amplitude (%)", fontsize=20)
    ax.set_ylabel("Density (a.u.)", fontsize=20)
    ax.set_yticks([])
    ax.tick_params(axis="x", which="major", labelsize=20)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.savefig(path)

def plot_time_spect_fit(
    tdata: np.ndarray,
    fdata: np.ndarray,
    fitdata: np.ndarray,
    ydata: np.ndarray,
    dwell_time: float,
    zeropad_size: int,
    path: str,
):
    """Plot the time domain and spectral domain fitting results."""

    plt.figure(figsize=(15, 5))
    plt.subplot(221)

    ax1 = plt.subplot(1, 3, 1)
    ax1.plot(tdata, abs(ydata))
    ax1.plot(tdata, abs(fitdata))

    ax1.legend(["broad time sig", "fit time sig"])

    # calculate fit spectral signal
    complex_fit_spect = dwell_time * np.fft.fftshift(np.fft.fft(fitdata, zeropad_size))
    spectral_signal = dwell_time * np.fft.fftshift(np.fft.fft(ydata, zeropad_size))
    ax2 = plt.subplot(1, 3, 2)
    ax2.plot(fdata, abs(spectral_signal), "*-")
    ax2.plot(fdata, abs(complex_fit_spect))
    ax2.set_xlim((-10000, 10000))
    ax2.legend(["spectral sig", "fit spect sig"])
    plt.savefig(path)


def plot_1d(x: np.ndarray, path: str = "tmp/1d_plot.png"):
    """Plot a 1D array.

    Args:
        x (np.ndarray): 1D array.
        path (str, optional): path to save the plot. Defaults to "tmp/1d_plot.png".
    """
    plt.figure(figsize=(15, 5))
    plt.plot(x)
    plt.savefig(path)


def plot_dynamics_all(
    area_dyn: np.ndarray,
    freq_dyn: np.ndarray,
    fwhmL_dyn: np.ndarray,
    phase_dyn: np.ndarray,
    t_dyn: np.ndarray,
    start_ind: int,
    end_ind: int,
    path: str = "tmp/dynamics.png",
):
    """Plot the non-detrended RBC, membrane, gas dynamics.

    Args:
        area_dyn (np.ndarray): the area of each component of shape (nframes, 3).
        freq_dyn (np.ndarray): the frequency of each component of shape (nframes, 3).
        fwhmL_dyn (np.ndarray): the fwhm of each component of shape (nframes, 3).
        phase_dyn (np.ndarray): the phase of each component of shape (nframes, 3).
        t_dyn (np.ndarray): the time points of the oscillations of shape (nframes, ).
        start_ind (int): start index of the oscillations being analyzed.
        end_ind (int): end index of the oscillations being analyzed.
        path (str, optional): path to save the plot. Defaults to "tmp/dynamics.png".
    """

    def set_plot_properties(
        ax,
        x_data,
        y_data,
        color,
        x_label=None,
        y_label=None,
        ylim=None,
        legend=None,
        xticks=[],
    ):
        ax.plot(x_data, y_data, color=color, linewidth=5)
        ax.set_xlim(0, x_data[-1])
        if ylim:
            ax.set_ylim(ylim)
        if x_label:
            ax.set_xlabel(x_label, fontsize=120)
        if y_label:
            ax.set_ylabel(y_label, fontsize=80)
        if legend:
            ax.legend(legend, loc="lower right", fontsize=30)

        ax.axvspan(x_data[start_ind], x_data[end_ind], facecolor="gray", alpha=0.5)
        ax.tick_params(axis="y", labelsize=42)
        ax.tick_params(axis="x", labelsize=42)
        ax.set_xticks(xticks)

    # Creating figures with 12 subplots
    fig, axs = plt.subplots(4, 3, figsize=(58, 42))
    fig.suptitle("Dynamics By Resonance", fontsize="200", fontweight="bold")

    fg_color = ["red", "green", "blue"]
    m3rd = np.arange(start_ind, end_ind, 1)
    plotlim = np.array([[2, 5, 1.5], [1.5, 0.5, 0.5], [5, 5, 0.5], [30, 6, 12]])

    for iComp in range(3):
        # Amplitude Plot
        set_plot_properties(
            ax=axs[0, iComp],
            x_data=t_dyn[:-5],
            y_data=area_dyn[:-5, iComp] / max(area_dyn[49:, 2]),
            color=fg_color[iComp],
            y_label="Amplitude" if iComp == 0 else None,
            ylim=[0, np.max(area_dyn[:, iComp]) / np.max(area_dyn[49:, 2])],
        )
        # Frequency or Chemical Shift Plot
        set_plot_properties(
            ax=axs[1, iComp],
            x_data=t_dyn,
            y_data=freq_dyn[:, iComp],
            color=fg_color[iComp],
            y_label="Shift (ppm)" if iComp == 0 else None,
            ylim=[
                np.mean(freq_dyn[m3rd, iComp]) - plotlim[1, iComp],
                np.mean(freq_dyn[m3rd, iComp]) + plotlim[1, iComp],
            ],
        )
        # FWHM Lorentzian Plot
        set_plot_properties(
            ax=axs[2, iComp],
            x_data=t_dyn,
            y_data=fwhmL_dyn[:, iComp],
            color=fg_color[iComp],
            y_label="FWHM (ppm)" if iComp == 0 else None,
            ylim=[
                np.mean(fwhmL_dyn[m3rd, iComp]) - plotlim[2, iComp],
                np.mean(fwhmL_dyn[m3rd, iComp]) + plotlim[2, iComp],
            ],
        )
        # Phase Plot
        set_plot_properties(
            ax=axs[3, iComp],
            x_data=t_dyn,
            y_data=phase_dyn[:, iComp],
            color=fg_color[iComp],
            y_label="Phase (deg)" if iComp == 0 else None,
            ylim=[
                np.mean(phase_dyn[m3rd, iComp]) - plotlim[3, iComp],
                np.mean(phase_dyn[m3rd, iComp]) + plotlim[3, iComp],
            ],
            xticks=np.arange(0, int(t_dyn[-1]), 2),
        )
        axs[3, iComp].tick_params(axis="x", labelsize=48)
    axs[0, 0].set_title("RBC", color="red", fontsize=120)
    axs[0, 1].set_title("Membrane", color="green", fontsize=120)
    axs[0, 2].set_title("Gas", color="blue", fontsize=120)
    plt.savefig(
        path,
        facecolor="w",
        edgecolor="w",
        orientation="portrait",
        format=None,
        transparent=False,
        bbox_inches="tight",
        pad_inches=0.01,
        metadata=None,
    )


def plot_oscillations_all(
    area_dyn_detrend: np.ndarray,
    area_dyn_fit: np.ndarray,
    area_dyn_indices_peaks: np.ndarray,
    freq_dyn_detrend: np.ndarray,
    freq_dyn_fit: np.ndarray,
    freq_dyn_indices_peaks: np.ndarray,
    fwhm_dyn_detrend: np.ndarray,
    fwhm_dyn_fit: np.ndarray,
    fwhm_dyn_indices_peaks: np.ndarray,
    phase_dyn_detrend: np.ndarray,
    phase_dyn_fit: np.ndarray,
    phase_dyn_indices_peaks: np.ndarray,
    t_dyn: np.ndarray,
    path: str = "tmp/oscillations.png",
):
    """Plot the oscillations of the area, frequency, fwhm, and phase.

    Args:
        area_dyn_detrend (np.ndarray): the detrended area of RBC component of
            shape (nframes, ).
        area_dyn_fit (np.ndarray): the fitted area of RBC component of shape
            (nframes, ).
        freq_dyn_detrend (np.ndarray): the detrended frequency of RBC component of
            shape (nframes, ).
        freq_dyn_fit (np.ndarray): the fitted frequency of RBC component of shape
            (nframes, ).
        fwhm_dyn_detrend (np.ndarray): the detrended fwhm of RBC component of shape
            (nframes, ).
        fwhm_dyn_fit (np.ndarray): the fitted fwhm of RBC component of shape
            (nframes, ).
        phase_dyn_detrend (np.ndarray): the detrended phase of RBC component of shape
            (nframes, ).
        phase_dyn_fit (np.ndarray): the fitted phase of RBC component of shape
            (nframes, ).
        t_dyn (np.ndarray): the time points of the oscillations of shape (nframes, ).
        path (str, optional): path to save the plot. Defaults to "tmp/oscillations.png".
    """

    def plot_subplot(
        ax: Axes,
        x_data: np.ndarray,
        detrend_data: np.ndarray,
        fitted_data: np.ndarray,
        peak_indices: np.ndarray,
        y_label: str,
        ylim: tuple[float, float],
    ):
        ax.plot(x_data, detrend_data, "--k", x_data, fitted_data, "r")
        ax.plot(x_data[peak_indices], detrend_data[peak_indices], "o", color="blue")
        ax.axhline(linewidth=1, color="k")
        ax.set_xticks(np.arange(np.ceil(t_dyn[0]), np.ceil(t_dyn[-1]), 2))
        ax.set_ylabel(y_label, fontsize=24)
        ax.set_xlim((t_dyn[0], t_dyn[-1]))
        ax.set_ylim(ylim)
        ax.tick_params(axis="y", labelsize=26)
        ax.xaxis.set_tick_params(labelsize=20)

    _, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize=(14, 15))

    plot_subplot(
        ax=ax1,
        x_data=t_dyn,
        detrend_data=area_dyn_detrend * 100,
        fitted_data=area_dyn_fit * 100,
        peak_indices=area_dyn_indices_peaks,
        y_label="Amplitude",
        ylim=(-20, 20),
    )
    plot_subplot(
        ax=ax2,
        x_data=t_dyn,
        detrend_data=freq_dyn_detrend,
        fitted_data=freq_dyn_fit,
        peak_indices=freq_dyn_indices_peaks,
        y_label="Shift (ppm)",
        ylim=(-0.5, 0.5),
    )
    plot_subplot(
        ax=ax3,
        x_data=t_dyn,
        detrend_data=fwhm_dyn_detrend,
        fitted_data=fwhm_dyn_fit,
        peak_indices=fwhm_dyn_indices_peaks,
        y_label="FWHM (ppm)",
        ylim=(-0.5, 0.5),
    )
    plot_subplot(
        ax=ax4,
        x_data=t_dyn,
        detrend_data=phase_dyn_detrend,
        fitted_data=phase_dyn_fit,
        peak_indices=phase_dyn_indices_peaks,
        y_label="Phase (deg)",
        ylim=(-10, 10),
    )
    ax1.set_title(
        "Detrended RBC Oscillations", color="red", fontsize=46, fontweight="bold"
    )
    ax4.set_xlabel("Time (s)", fontsize=40)
    plt.savefig(
        path,
        facecolor="w",
        edgecolor="w",
        orientation="portrait",
        format=None,
        transparent=False,
        bbox_inches="tight",
        pad_inches=0.01,
    )


def plot_static_spectra(
    fit_obj: nmr_timefit.NMR_TimeFit,
    freq_center: float,
    dwell_time: float,
    path: str = "tmp/static_spectra.png",
):
    """Plot the static spectroscopy fit and the components.

    Args:
        fit_obj (nmr_timefit.NMR_TimeFit): the fit object.
        freq_center (float): the center frequency in MHz.
        dwell_time (float): the dwell time in seconds.
        path (str, optional): path to save the plot.
    """

    # Extracting data from fit_obj
    ref_freq = fit_obj.freq[-1]
    fdata = (np.linspace(-0.5, 0.5, len(fit_obj.tdata) * 2 + 1) / dwell_time)[0:-1]

    ppm = (fdata - constants.XENON_SHIFT * freq_center - ref_freq) / freq_center
    ppm_shift = (fit_obj.freq[-1] - ref_freq) / freq_center
    tplot = np.linspace(fit_obj.tdata[0], 2 * fit_obj.tdata[-1], len(fdata))
    fit_ref_obj = nmr_timefit.NMR_TimeFit(
        ydata=np.ones(fit_obj.tdata.shape),
        tdata=fit_obj.tdata,
        area=constants.REFERENCEFIT.AREA,
        freq=constants.REFERENCEFIT.FREQ * freq_center + ref_freq,
        fwhmL=constants.REFERENCEFIT.FWHM * freq_center,
        fwhmG=constants.REFERENCEFIT.FWHMG * freq_center,
        phase=constants.REFERENCEFIT.PHASE + fit_obj.phase[1],
        method="voigt",
        line_broadening=0,
        zeropad_size=np.size(fit_obj.tdata),
    )
    spectrum_ref_components = dwell_time * np.fft.fftshift(
        np.fft.fft(fit_ref_obj.get_time_function_components(tplot), axis=0)
    )
    spectrum_components = dwell_time * np.fft.fftshift(
        np.fft.fft(fit_obj.get_time_function_components(tplot), axis=0)
    )
    spectrum = dwell_time * np.fft.fftshift(
        np.fft.fft(fit_obj.get_time_function(fit_obj.tdata), axis=0)
    )
    # Creating figure with 3 subplots
    _, (ax1, ax2, ax3) = plt.subplots(3, figsize=(12, 9))
    ax1.plot(
        ppm - ppm_shift,
        np.abs(spectrum_ref_components)
        / np.sum(np.max(abs(spectrum_ref_components[:, 2]))),
        "k:",
        linewidth=2.5,
    )

    colors = ["blue", "red", "green"]
    for index in range(3):
        ax1.plot(
            ppm,
            abs(spectrum_components[:, index])
            / np.max(abs(spectrum_components[:, -1])),
            color=colors[index],
            linewidth=2.5,
        )

    # ax1.set_xlabel('Chemical Shift (ppm)', fontsize=9)
    ax1.set_ylabel("Component Intensity", fontsize=20, fontweight="bold")
    ax1.set_xlim(150, 250)
    ax1.set_ylim(0, 1)
    ax1.invert_xaxis()
    ax1.set_title("Static Spectroscopy", color="blue", fontsize=48, fontweight="bold")

    ax2.plot(
        (fit_obj.f - ref_freq) / freq_center,
        np.real(fit_obj.spectral_signal),
        ".k",
        markersize=10,
        label="Measured",
    )
    ax2.plot(
        (fit_obj.f - ref_freq) / freq_center,
        np.real(spectrum),
        "-g",
        linewidth=2.5,
        label="Fitted",
    )
    ax2.plot(
        (fit_obj.f - ref_freq) / freq_center,
        np.real(spectrum - fit_obj.spectral_signal),
        ".r",
        markersize=10,
        label="Residual",
    )
    ax2.set_ylabel("Real", fontsize=25, fontweight="bold")
    ax2.set_xlim(150, 250)
    ax2.invert_xaxis()
    ax2.legend(loc="lower right")

    ax3.plot(
        (fit_obj.f - ref_freq) / freq_center,
        np.imag(fit_obj.spectral_signal),
        ".k",
        markersize=10,
    )
    ax3.plot(
        (fit_obj.f - ref_freq) / freq_center,
        np.imag(spectrum),
        "-g",
        linewidth=2.5,
    )
    ax3.plot(
        (fit_obj.f - ref_freq) / freq_center,
        np.imag(spectrum - fit_obj.spectral_signal),
        ".r",
        markersize=10,
    )
    ax3.set_xlabel("Chemical Shift (ppm)", fontsize=32, fontweight="bold")
    ax3.set_ylabel("Imaginary", fontsize=25, fontweight="bold")
    ax3.set_xlim(150, 250)
    ax3.invert_xaxis()

    ax1.xaxis.set_tick_params(labelsize=25)
    ax1.yaxis.set_tick_params(labelsize=25)
    ax2.xaxis.set_tick_params(labelsize=25)

    ax2.yaxis.set_tick_params(labelsize=25)
    ax3.xaxis.set_tick_params(labelsize=25)
    ax3.yaxis.set_tick_params(labelsize=25)
    plt.tight_layout()

    plt.savefig(
        path,
        facecolor="w",
        edgecolor="w",
        orientation="portrait",
        transparent=False,
        bbox_inches="tight",
        pad_inches=0.01,
    )


def plot_dynamic_snr(
    x: np.ndarray,
    t: np.ndarray,
    start_ind: int,
    end_ind: int,
    path: str = "tmp/dynamic_snr.png",
):
    """Plot the SNR of each FID.

    Args:
        x (np.ndarray): SNR of each FID of shape (nframes, ).
        t (np.ndarray): time of each FID of shape (nframes, ).
        start_ind (int): start index of the oscillations being analyzed.
        end_ind (int): end index of the oscillations being analyzed.
        path (str, optional): path to save the plot. Defaults to "tmp/dynamic_snr.png".
    """

    plt.figure(figsize=(8, 3))
    plt.plot(t, x)
    plt.ylim(10, 30)
    plt.xlim(0, t[-1])
    plt.yticks(np.arange(10, 31, 5))
    plt.text(
        3,
        27,
        "Mean SNR: " + str(np.round(np.mean(x), 1)),
        color="red",
        fontsize=20,
        fontweight="bold",
        ha="center",
        va="center",
        bbox=dict(boxstyle="square", facecolor="wheat", alpha=0.5),
    )
    plt.ylabel("SNR", fontsize=14, fontweight="bold")
    plt.xlabel("Time in Seconds", fontsize=14, fontweight="bold")
    plt.title("Dynamic SNR", fontsize=28, fontweight="bold")
    plt.axvspan(t[start_ind], t[end_ind], alpha=0.3, color="grey")
    plt.tick_params(labelsize=10)
    plt.savefig(
        path,
        facecolor="w",
        edgecolor="w",
        orientation="portrait",
        transparent=False,
        bbox_inches="tight",
        pad_inches=0.01,
        metadata=None,
    )


def plot_montage_phase(
    image: np.ndarray, path: str, index_start: int, index_skip: int = 1
):
    """Plot a montage of the phase image in grey scale.

    Will make a montage of 2x8 of the image in grey scale and save it to the path.
    Assumes the image is of shape (x, y, z) where there are at least 16 slices.
    Otherwise, will plot all slices.

    Args:
        image (np.ndarray): gray scale phase image between -180 and 180 deg to plot of shape (x, y, z)
        path (str): path to save the image.
        index_start (int): index to start plotting from.
        index_skip (int, optional): indices to skip. Defaults to 1.
    """
    # plot the montage
    index_end = index_start + index_skip * 16
    # stack the image to make it 4D (x, y, z, 3)
    image = np.stack((image, image, image), axis=-1)
    montage = make_montage(
        image[:, :, index_start:index_end:index_skip, :], n_slices=16
    )

    plt.figure()
    plt.imshow(montage)

    plt.axis("off")
    plt.savefig(path, transparent=True, bbox_inches="tight", pad_inches=-0.05, dpi=300)
    plt.clf()
    plt.close()


def plot_complex_montage(image, path, index_start, index_skip):
    """
    Plot a montage of a 3D complex-valued image.

    Args:
        image (np.ndarray): 3D complex-valued image of shape (x, y, z)
        path (str): path to save the montage image.
        index_start (int): index to start plotting from.
        index_end (int): index to end plotting.
    """
    index_end = index_start + index_skip * 16
    # Extract the magnitude and phase of the complex image
    magnitude = np.abs(image)
    phase = np.angle(image, deg=True)

    # Normalize magnitude to [0, 1] for display
    magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min())

    # Extract slices based on provided indices
    magnitude_slices = magnitude[:, :, index_start:index_end:index_skip]
    phase_slices = phase[:, :, index_start:index_end:index_skip]

    # Create a custom colormap that uses magnitude for intensity and phase for color
    colors = plt.cm.hsv((phase_slices + 180) / 360.0)  # Convert phase to [0, 1] range
    colored_slices = colors[..., :3] * magnitude_slices[..., np.newaxis]

    # Prepare the montage grid
    montage_image = np.zeros((2 * image.shape[0], 8 * image.shape[1], 3))

    # Fill the montage grid with slices
    for idx in range(colored_slices.shape[2]):
        row_idx = idx // 8
        col_idx = idx % 8
        montage_image[
            row_idx * image.shape[0] : (row_idx + 1) * image.shape[0],
            col_idx * image.shape[1] : (col_idx + 1) * image.shape[1],
            :,
        ] = colored_slices[:, :, idx]

    # Plot and save the montage
    plt.figure(figsize=(15, 5))
    plt.imshow(montage_image)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()


def plot_histogram_ventilation(data: np.ndarray, path: str):
    """Plot histogram of ventilation.

    Args:
        data (np.ndarray): data to plot histogram of.
        path (str): path to save the image.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    data = data.flatten()
    # normalize the 99th percentile
    data = data / np.percentile(data, 99)
    data[data > 1] = 1
    weights = np.ones_like(data) / float(len(data))
    # plot histogram
    _, bins, _ = ax.hist(
        data,
        bins=50,
        color=(0.4196, 0.6824, 0.8392),
        weights=weights,
        edgecolor="black",
    )

    # plot healthy reference line
    refer_fit = np.array([0.0407, 0.619, 0.196])
    normal = refer_fit[0] * np.exp(-(((bins - refer_fit[1]) / refer_fit[2]) ** 2))
    ax.plot(bins, normal, "--", color="k", linewidth=4)
    ax.set_ylabel("Fraction of Total Pixels", fontsize=35)
    # set plot parameters
    plt.xlim((0, 1))
    plt.ylim((0, 0.06))
    plt.rc("axes", linewidth=4)
    # set ticks
    xticks = [0.0, 0.5, 1.0]
    yticks = [0.02, 0.04, 0.06]
    plt.xticks(xticks, ["{:.0f}".format(x) for x in xticks], fontsize=40)
    plt.yticks(yticks, ["{:.2f}".format(x) for x in yticks], fontsize=40)
    fig.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_image_slice(
    image: np.ndarray,
    path: str,
    slice_index: int,
    vmax: Optional[float] = None,
    vmin: Optional[float] = None,
    cmap: str = "gray",
):
    """Plot the image slice."""
    plt.figure()
    plt.imshow(image[:, :, slice_index], cmap=cmap, vmin=vmin, vmax=vmax)
    plt.axis("off")
    plt.savefig(path, transparent=True, bbox_inches="tight", pad_inches=-0.05, dpi=300)
    plt.clf()
    plt.close()
