"""Module for segmentation model inference.

@author: ZiyiW Now Sup
"""
import os

import numpy as np
from absl import app, flags
from scipy.ndimage import (zoom, binary_fill_holes, iterate_structure, binary_dilation,
                           generate_binary_structure, binary_erosion)
from typing import Optional, Literal

from models.model_vnet import vnet
from utils import constants, img_utils, io_utils

# define flags
FLAGS = flags.FLAGS

flags.DEFINE_string("image_type", "vent", "either ute or vent for segmentation")
flags.DEFINE_string("nii_filepath", "", "nii image file path")

def threshold_mask(arr: np.ndarray,
    percentile: float = 80.0,
    morph: Optional[Literal["erode", "dilate"]] = None,  # None | 'erode' | 'dilate'
    radius: int = 1,                           # structuring element radius (>=1 to have effect)
    iterations: int = 1,                         # how many times to apply morph op
    fill_holes: bool = True,                      # ensure no interior holes
    connectivity: Optional[int] = 2                 # neighborhood connectivity; None -> auto
) -> np.ndarray:
    """
    Create a boolean mask from `arr` where elements strictly greater than the given
    percentile are True, with optional erosion/dilation and 3D (nD) hole filling.

    Parameters
    ----------
    arr : np.ndarray
        Input array (2D, 3D, or nD). NaNs are ignored for percentile computation.
    percentile : float, default=80.0
        Percentile threshold. Elements strictly '>' this value become True.
    morph : {'erode', 'dilate', None}, default=None
        Optional morphological operation on the mask.
    radius : int, default=1
        “Radius” of the structuring element. Uses an nD connectivity structure
        expanded by `radius`. Must be >= 1 to have effect.
    iterations : int, default=1
        Number of times to apply the morphological operation.
    fill_holes : bool, default=True
        If True, performs nD hole filling to remove interior cavities.
    connectivity : int or None, default=2
        Connectivity for the structuring element (1..arr.ndim). For full connectivity
        use `connectivity=arr.ndim`. If None, defaults to 2 (or to arr.ndim if arr.ndim < 2).

    Returns
    -------
    mask : np.ndarray (bool)
        Boolean mask of the same shape as `arr`.
    """
    ##Standardize image
    arr = np.abs(arr)
    arr = 255 * (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

    thr = float(np.nanpercentile(arr, percentile))

    # Binary mask: greater than equal to the threshold
    mask = arr >= thr
    # Fill interior holes (first pass)
    if fill_holes:
        mask = binary_fill_holes(mask)
    # Optional morphological operation
    morph = (morph or "").lower()
    if morph in ("erode", "dilate") and iterations > 0 and radius >= 1:
        # Determine connectivity (cap within valid range)
        if connectivity is None:
            conn = min(max(1, 2), mask.ndim)  # default to 2 when possible
        else:
            conn = int(connectivity)
            conn = min(max(1, conn), mask.ndim)
        struct = generate_binary_structure(rank=mask.ndim, connectivity=conn)
        if radius > 1:
            struct = iterate_structure(struct, radius)
        if morph == "erode":
            mask = binary_erosion(mask, structure=struct, iterations=iterations, border_value=0)
        else:  # 'dilate'
            mask = binary_dilation(mask, structure=struct, iterations=iterations, border_value=0)
        # Re-fill holes to guarantee no interior holes after morph
        if fill_holes:
            mask = binary_fill_holes(mask)

    return mask


def predict(
    image: np.ndarray,
    image_type: str = constants.ImageType.VENT.value,
    erosion: int = 0,
) -> np.ndarray:
    """Generate a segmentation mask from the proton or ventilation image.

    Args:
        image: np.nd array of the input image to be segmented.
        image_type: str of the image type ute or vent.
    Returns:
        mask: np.ndarray of type bool of the output mask.
    """
    # get shape of the image
    img_h, img_w, _ = np.shape(image)
    # reshaping image for segmentation
    if img_h == 64 and img_w == 64:
        print("Reshaping image for segmentation")
        image = zoom(abs(image), [2, 2, 2])
    elif img_h == 128 and img_w == 128:
        pass
    else:
        raise ValueError("Segmentation Image size should be 128 x 128 x n")

    if image_type == constants.ImageType.VENT.value:
        model = vnet(input_size=(128, 128, 128, 1))
        weights_dir_current = "./models/weights/model_ANATOMY_VEN.h5"
    else:
        raise ValueError("image_type must be ute or vent")

    # Load model weights
    model.load_weights(weights_dir_current)

    if image_type == constants.ImageType.VENT.value:
        image = img_utils.standardize_image(image)
    else:
        raise ValueError("Image type must be ute or vent")
    # Model Prediction
    image = image[None, ...]
    image = image[..., None]
    mask = model.predict(image)
    # Making mask binary
    mask = mask[0, :, :, :, 0]
    mask[mask > 0.5] = 1
    mask[mask < 1] = 0
    # erode mask
    if erosion > 0:
        mask = img_utils.erode_image(mask, erosion)
    return mask.astype(bool)


def main(argv):
    """Run CNN model inference on ute or vent image."""
    image = io_utils.import_nii(FLAGS.nii_filepath)
    image_type = FLAGS.image_type
    mask = predict(image, image_type)
    export_path = os.path.join(os.path.dirname(FLAGS.nii_filepath), "mask.nii")
    io_utils.export_nii(image=mask.astype("float64"), path=export_path)


if __name__ == "__main__":
    app.run(main)
