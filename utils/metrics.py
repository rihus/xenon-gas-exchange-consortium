"""Metrics for evaluation."""

import math
import sys
from datetime import datetime
from typing import Optional

import numpy as np

from scipy.ndimage import binary_dilation

from utils import constants

sys.path.append("..")

def get_snr_tail(
    area: np.ndarray, ydata: np.ndarray, fitdata: np.ndarray
) -> np.ndarray:
    """Calculate the historical `tail of fit residuals` SNR.

    Args:
        area (np.ndarray): area of the fit peaks.
        ydata (np.ndarray): time-domain data.
        fitdata (np.ndarray): time-domain fit.
    """
    residual = fitdata - ydata
    std_ = np.std(residual[-np.round(len(residual) / 4).astype(int) : -1])
    return area / std_


def oscillation_amplitude(amplitude: float) -> float:
    """Return the peak to peak oscillation amplitude.

    Args:
        amplitude (float): the amplitude obtained from fitting the oscillation data
        to A * sin(w * t + phi)
    """
    return amplitude * 2


def oscillation_amplitude_percent(amplitude: float) -> float:
    """Return the peak to peak oscillation amplitude percentage.

    Args:
        amplitude (float): the amplitude obtained from fitting the oscillation data
        to A * sin(w * t + phi)
    """
    return 100 * amplitude * 2


def peaks_to_amplitude_percent(peaks: np.ndarray) -> float:
    """Return the peak to peak oscillation amplitude percentage.

    Args:
        peaks (np.ndarray): the peaks of the oscillation data
    """
    return (np.median(peaks[peaks > 0]) - np.median(peaks[peaks < 0])) * 100


def peaks_to_amplitude(peaks: np.ndarray) -> float:
    """Return the peak to peak oscillation amplitude.

    If there are no positive or negative peaks, return 0.
    Args:
        peaks (np.ndarray): the peaks of the oscillation data
    """
    high = np.median(peaks[peaks > 0])
    low = np.median(peaks[peaks < 0])
    if np.isnan(high) or np.isnan(low):
        return 0
    else:
        return np.median(peaks[peaks > 0]) - np.median(peaks[peaks < 0])


def mean_snr(snr_arr: np.ndarray) -> float:
    """Return the mean SNR of an array of SNRs.

    Args:
        snr_arr (np.ndarray): array of SNRs
    """
    return np.mean(snr_arr)


def heart_rate(omega: float) -> float:
    """Return the heart rate in beats per minute.

    Args:
        omega (float): the angular frequency of the oscillation
    """
    return omega * 60 / (2 * np.pi)


def _get_dilation_kernel(x: int) -> int:
    """Get dilation kernel for binary dilation in 1-dimension."""
    return int((math.ceil(x * 0.025) * 2 + 1))


def snr(image: np.ndarray, mask: np.ndarray, window_size: int = 8):
    """Calculate SNR using sliding windows.

    Args:
        image (np.ndarray): 3-D array of image data.
        mask (np.ndarray): 3-D array of mask data.
        window_size (int): size of the sliding window for noise calculation.
            Defaults to 8.
    Returns:
        Tuple of SNR and Rayleigh SNR and image noise
    """
    # convert mask to boolean
    mask = mask.astype(bool)
    shape = np.shape(image)
    # dilate the mask to analyze noise area away from the signal
    kernel_shape = (
        _get_dilation_kernel(shape[0]),
        _get_dilation_kernel(shape[1]),
        _get_dilation_kernel(shape[2]),
    )
    dilate_struct = np.ones((kernel_shape))
    noise_mask = binary_dilation(mask, dilate_struct).astype(bool)

    noise_temp = np.copy(image)
    noise_temp[noise_mask] = np.nan
    # set up for using mini noise cubes through the image and calculate std for noise
    n_noise_vox = window_size * window_size * window_size
    mini_vox_std = 0.75 * n_noise_vox  # minimul number of voxels to calculate std

    stepper = 0
    total = 0
    std_dev_mini_noise_vol = []

    for ii in range(0, int(shape[0] / window_size)):
        for jj in range(0, int(shape[1] / window_size)):
            for kk in range(0, int(shape[2] / window_size)):
                mini_cube_noise_dist = noise_temp[
                    ii * window_size : (ii + 1) * window_size,
                    jj * window_size : (jj + 1) * window_size,
                    kk * window_size : (kk + 1) * window_size,
                ]
                mini_cube_noise_dist = mini_cube_noise_dist[
                    ~np.isnan(mini_cube_noise_dist)
                ]
                # only calculate std for the noise when it is long enough
                if len(mini_cube_noise_dist) > mini_vox_std:
                    std_dev_mini_noise_vol.append(np.std(mini_cube_noise_dist, ddof=1))
                    stepper = stepper + 1
                total = total + 1

    image_noise = np.median(std_dev_mini_noise_vol)
    image_signal = np.average(image[mask])

    snr_ = image_signal / image_noise
    return snr_, snr_ * 0.66, image_noise


def mse(image1: np.ndarray, image2: np.ndarray) -> float:
    """Calculate mean squared error between two images.

    Args:
        image1: np.ndarray
        image2: np.ndarray
    """
    return np.mean((image1 - image2) ** 2)


def nrmse(image1: np.ndarray, image2: np.ndarray) -> float:
    """Calculate the Normalized Root Mean Squared Error (NRMSE) between images.

    NRMSE is a normalized version of the root mean squared error that ranges from 0 to 1
    (or 0 to 100%) and is useful when comparing different datasets or error measures
    against each other.

    Parameters:
    x (numpy.ndarray): The first numpy array.
    y (numpy.ndarray): The second numpy array, must be the same size as x.

    Returns:
    float: The NRMSE value between the two arrays.

    Raises:
    ValueError: If the input arrays do not have the same shape.
    """

    if image1.shape != image2.shape:
        raise ValueError("Input arrays must have the same shape.")

    # Calculate RMSE
    mse_ = np.mean((image1 - image2) ** 2)
    rmse = np.sqrt(mse_)

    # Normalize RMSE
    range_x = np.max(image1) - np.min(image1)
    n_rmse = rmse / range_x if range_x != 0 else float("inf")

    return n_rmse


def inflation_volume(mask: np.ndarray, fov: float) -> float:
    """Calculate the inflation volume of isotropic 3D image.

    Args:
        mask: np.ndarray thoracic cavity mask.
        fov: float field of view in cm
    Returns:
        Inflation volume in L.
    """
    return (
        np.sum(mask) * fov**3 / np.shape(mask)[0] ** 3
    ) / constants.FOVINFLATIONSCALE3D

def process_date() -> str:
    """Return the current date in YYYY-MM-DD format."""
    now = datetime.now()
    return now.strftime("%Y-%m-%d")

def bin_percentage(
    image: np.ndarray, bins: np.ndarray, mask: Optional[np.ndarray] = None
) -> float:
    """Get the percentage of voxels in the given bins.

    Args:
        image: np.ndarray binned image. Assumes that the values in the image are
            integers representing the bin number. Bin 0 is the region outside the mask
            and Bin 1 is the lowest bin, etc.
        bins: np.ndarray list of bins to include in the percentage calculation.
    """
    if mask is None:
        return 100 * np.sum(np.isin(image, bins)) / np.sum(image > 0)
    else:
        return 100 * np.sum(np.isin(image[mask], bins)) / np.sum(mask)


def mean(image: np.ndarray, mask: np.ndarray) -> float:
    """Get the mean of the image.

    Args:
        image: np.ndarray. The image.
        mask: np.ndarray. mask of the region of interest.()
    Returns:
        Mean of the image.
    """
    return np.mean(image[mask])


def negative_voxels_percentage(image: np.ndarray, mask: np.ndarray) -> float:
    """Get the percentage voxels of image inside mask that are negative.

    Args:
        image: np.ndarray. The image.
        mask: np.ndarray. mask of the region of interest.
    Returns:
        Percentage of voxels in the image that are negative.
    """
    return 100 * np.sum(image[mask] < 0) / np.sum(mask)


def median(image: np.ndarray, mask: np.ndarray) -> float:
    """Get the median of the image.

    Args:
        image: np.ndarray. The image.
        mask: np.ndarray. mask of the region of interest.
    Returns:
        Median of the image.
    """
    return np.median(image[mask])


def std(image: np.ndarray, mask: np.ndarray) -> float:
    """Get the standard deviation of the image.

    Args:
        image: np.ndarray. The image.
        mask: np.ndarray. mask of the region of interest.
    Returns:
        Standard deviation of the image.
    """
    return np.std(image[mask])


def dlco(
    image_gas: np.ndarray,
    image_membrane: np.ndarray,
    image_rbc: np.ndarray,
    mask: np.ndarray,
    mask_vent: np.ndarray,
    fov: float,
    membrane_mean: float = 0.736,
    rbc_mean: float = 0.471,
) -> float:
    """Get the DLCO of the image.

    Reference: https://journals.physiology.org/doi/epdf/10.1152/japplphysiol.00702.2020
    Args:
        image_gas: np.ndarray. The ventilation image.
        image_membrane: np.ndarray. The membrane image.
        img_rbc: np.ndarray. The RBC image.
        mask: np.ndarray. thoracic cavity mask.
        mask_vent: np.ndarray. mask of the non-VDP region.
        fov: float. field of view in cm.
        membrane_mean: float. The mean membrane in healthy subjects.
        rbc_mean: float. The mean RBC in healthy subjects.
    """
    return kco(
        image_membrane, image_rbc, mask_vent, membrane_mean, rbc_mean
    ) * alveolar_volume(image_gas, mask, fov)


def alveolar_volume(image: np.ndarray, mask: np.ndarray, fov: float) -> float:
    """Get the alveolar volume of the image.

    Reference: https://journals.physiology.org/doi/epdf/10.1152/japplphysiol.00702.2020
    Args:
        image: np.ndarray. The binned ventilation image.
        mask: np.ndarray. thoracic cavity mask.
        fov: float. field of view in cm.
    Returns:
        Alveolar volume in L.
    """
    return (
        constants.VA_ALPHA
        * inflation_volume(mask, fov)
        * (1.0 - bin_percentage(image, np.asarray([1]), mask) / 100)
    )


def kco(
    image_membrane: np.ndarray,
    image_rbc: np.ndarray,
    mask: np.ndarray,
    membrane_mean: float = 0.736,
    rbc_mean: float = 0.471,
) -> float:
    """Get the KCO of the image.

    Reference: https://journals.physiology.org/doi/epdf/10.1152/japplphysiol.00702.2020
    Args:
        image_membrane: np.ndarray. The membrane image.
        img_rbc: np.ndarray. The RBC image.
        mask: np.ndarray. mask of non-VDP region.
        membrane_mean: float. The mean membrane in healthy subjects.
        rbc_mean: float. The mean RBC in healthy subjects.
    """
    membrane_rel = mean(image_membrane, mask) / membrane_mean
    rbc_rel = mean(image_rbc, mask) / rbc_mean
    membrane_rel = 1.0 / membrane_rel if membrane_rel > 1 else membrane_rel
    return 1 / (
        1 / (constants.KCO_ALPHA * membrane_rel) + 1 / (constants.KCO_BETA * rbc_rel)
    )


def relative_vc(
    subject_age: int,
    subject_sex: int,
    subject_height: float,
) -> float:
    """Get the relative capillary blood volume.

    Args:
        subject_age: int. Age of the subject
        subject_sex: int. 1 if female, 2 if male
        subject_height: float. Height of subject in cm
    """
    va = (
        constants.VA_ALPHA_MUNKHOLM
        * float(constants.StatsIOFields.INFLATION)
        * (1 - (float(constants.StatsIOFields.VENT_DEFECT_PCT) / 100))
    )
    if subject_sex == 1:
        predicted = -13.8 + (0.527 * subject_height) - (0.00421 * (subject_age ^ 2))
        estimated = (
            va
            * constants.KCO_BETA_MUNKHOLM
            * constants.THETA_INV_FEMALE
            * (float(constants.StatsIOFields.RBC_MEAN) / constants.RBC_REF)
        )
        return estimated / predicted
    elif subject_sex == 2:
        predicted = -23.8 + (0.645 * subject_height) - (0.00547 * (subject_age ^ 2))
        estimated = (
            va
            * constants.KCO_BETA_MUNKHOLM
            * constants.THETA_INV_MALE
            * (float(constants.StatsIOFields.RBC_MEAN) / constants.RBC_REF)
        )
        return estimated / predicted
    else:
        return 0.0


def relative_vc_map(
    subject_age: int,
    subject_sex: int,
    subject_height: float,
    rbc_img: np.ndarray,
    mask: np.ndarray,
):
    """Get a map of the voxel-wise relative capillary blood volume.

    Args:
        subject_age: int. Age of the subject
        subject_sex: int. 1 if female, 2 if male
        subject_height: float. Height of subject in cm
        rbc_img: np.ndarray. RBC image normalized to gas image
        mask: np.ndarray. Mask of non-VDP region.
    """
    va = constants.VA_ALPHA_MUNKHOLM * constants.VOXEL_SIZE

    if subject_sex == 1:
        predicted = (
            -13.8 + (0.527 * subject_height) - (0.00421 * (subject_age**2))
        ) / (np.count_nonzero(mask))
        print(predicted)
        print(np.count_nonzero(mask))
        estimated = (
            va
            * constants.KCO_BETA_MUNKHOLM
            * constants.THETA_INV_FEMALE
            * np.divide(np.abs(rbc_img), constants.RBC_REF)
        )
        print(np.sum(estimated))
        return np.divide(estimated, predicted)
    elif subject_sex == 2:
        predicted = (
            -23.8 + (0.645 * subject_height) - (0.00547 * (subject_age**2))
        ) / (np.count_nonzero(mask))
        print(predicted)
        print(np.count_nonzero(mask))
        estimated = (
            va
            * constants.KCO_BETA_MUNKHOLM
            * constants.THETA_INV_MALE
            * np.divide(np.abs(rbc_img), constants.RBC_REF)
        )
        print(np.sum(estimated))
        return np.divide(estimated, predicted)
    else:
        return 0.0
