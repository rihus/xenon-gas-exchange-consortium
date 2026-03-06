"""Utility functions for signal processing."""
import sys
from typing import Optional, Tuple

import numpy as np
import pywt
from scipy import optimize, signal, stats

import matlab_engine
from utils import constants
sys.path.append("..")


def _movmean(x: np.ndarray, n: int) -> np.ndarray:
    """Compute moving mean of x over n points.

    Args:
        x (np.ndarray): input data of shape (n,)
        n (int): number of points to average over

    Returns:
        np.ndarray: moving mean of shape (n,)
    """
    return np.convolve(x, np.ones((n,)) / n, mode="same")


def _getxygrid(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Get x and y grid.

    This is a lazy implementation of the Matlab function getxygrid in sethandles.m
    Args:
        x (np.ndarray): x data of shape (n,) must have at least 2 elements
            x cannot also have repeated x entries.
        y (np.ndarray): y data of shape (n,) must have at least 2 elements
    Returns:
        Tuple of x and y grid respectively
    """
    # check number of data points to be > 2
    assert len(x) > 2, "x must have at least 2 elements"
    # sort data points to be in order of increasing x
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]

    return (x, y)


def _sinnstart(x: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    """Get starting points fit for sum of n sine functions.

    Computes a starting for the parameters of a sum of n sine functions. By
    Running the y data through a fft, and then locating peaks in results, we
    can find the starting value of the frequency of each sine function 'b'
    in the function y = a*sin(b*x+c). Because a phase-shifted sine function
    is separable and can be converted to a sum of sine and cosine functions,
    starting values for amplitude 'a' and phase 'c' can be found.

    Returns:
        Starting values for the parameters of a sum of n sine functions of
            shape (n * 3, )
    """
    lenx = len(x)
    x, y = _getxygrid(x, y)
    # if data size is too small, cannot find starting values
    if len(x) < 2:
        return np.random.rand(n * 3)
    # loop for sum of sines functions
    start = np.zeros(3 * n)
    oldpeaks = np.array([])
    freqs = np.zeros(n)
    res = np.copy(y)
    for i in range(n):
        # apply fft to the current residuals
        fy = np.fft.fft(res)
        # omit frequencies already used
        if len(oldpeaks) > 0:
            fy[oldpeaks] = 0
        # get starting value for frequency using fft peaks
        maxloc = np.argmax(np.abs(fy[np.arange(0, np.floor(lenx / 2)).astype(int)]))
        np.append(oldpeaks, maxloc)
        w = 2 * np.pi * max((0.5, maxloc)) / (x[-1] - x[0])
        freqs[i] = w
        # compute fourier terms using all frequencies we have so far
        X = np.zeros((lenx, 2 * (i + 1)))
        for j in range(i + 1):
            X[:, 2 * j] = np.sin(freqs[j] * x)
            X[:, 2 * j + 1] = np.cos(freqs[j] * x)
        # fit these terms to get the non-frequency starting values
        ab = np.linalg.lstsq(X, y, rcond=None)[0]
        if i < n:
            res = y - np.matmul(X, ab)
    # all frequencies found, now compute starting values from all
    # frequencies and the corresponding coefficients.
    for i in range(n):
        start[3 * i] = np.sqrt(ab[2 * i] ** 2 + ab[2 * i + 1] ** 2)
        start[3 * i + 1] = freqs[i]
        start[3 * i + 2] = np.arctan2(ab[2 * i + 1], ab[2 * i])
    return start


def _sinbounds(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Get bounds for sum of n sine functions.

    Upper bounds are inf. Lower bounds are [-inf, 0, -inf] repeated n times.

    Args:
        n (int): number of sine functions
    Returns:
        Tuple of lower and upper bounds respectively of length n * 3
    """
    return (
        np.tile([-np.inf, 0, -np.inf], n),
        np.tile([np.inf, np.inf, np.inf], n),
    )


def boxcox(data: np.ndarray):
    """Apply box cox transformation on data.

    Args:
        data (np.ndarray): data to be transformed of shape (n,)
    Returns:
        Tuple of transformed data and box cox lambda
    """
    return stats.boxcox(data)


def inverse_boxcox(
    boxcox_lambda: float, data: np.ndarray, scale_factor: float
) -> np.ndarray:
    """Apply inverse box cox transformation on data.

    Args:
        boxcox_lambda (float): box cox lambda
        data (np.ndarray): data to be transformed of shape (n,)
        scale_factor (float): scale factor to be applied to the data
    """
    return np.power(boxcox_lambda * data + 1, 1 / boxcox_lambda) - scale_factor


def remove_gasphase_contamination(
    data_dissolved: np.ndarray,
    data_gas: np.ndarray,
    sample_time: float,
    freq_gas_acq_diss: float,
    phase_gas_acq_diss: float,
    area_gas_acq_diss: float,
    fa_gas: float,
) -> np.ndarray:
    """Remove gas phase contamination in dissolved k-space.

    Takes gas phase k-space and modifies it using NMR fits and gas phase k0
    to produce the expected gas phase contamination k-space data which is
    then removed from the initial contaminated dissolved phase k-space.

    Args:
        data_dissolved (np.ndarray): dissolved k-space data of shape
            (n_projections, n_points)
        data_gas (np.ndarray): gas phase k-space data of shape
            (n_projections, n_points)
        sample_time (float): dwell time in seconds.
        freq_gas_acq_diss (float): gas frequency offset in dissolved
            spectra acquisition in Hz.
        phase_gas_acq_diss (float): gas phase in dissolved spectra acquisition.
            in degrees.
        area_gas_acq_diss (float): gas area in dissolved spectra acquisition.
        fa_gas (float): gas flip angle in degrees.
    Returns:
        Gas phase corrected dissolved k-space data of shape (n_projections, n_points)
    Author: Matt Willmering
    Paper: https://pubmed.ncbi.nlm.nih.gov/33665905/
    """
    # step 0: calculate parameters
    arr_t = sample_time * np.arange(data_dissolved.shape[1])
    # step 1: modulate contamination (gas) to dissolved frequency - first order
    # phase approximation
    phase_shift1 = 2 * np.pi * freq_gas_acq_diss * arr_t  # calculate phase accumulation
    contamination_kspace1 = data_gas * np.exp(1j * phase_shift1)
    # step 2: zero order phase shift of contamination estimation
    phase_shift2 = phase_gas_acq_diss - 180 / np.pi * np.mean(np.angle(data_gas[:, 0]))
    contamination_kspace2 = contamination_kspace1 * np.exp(
        1j * np.pi / 180 * phase_shift2
    )
    # step 3: scale contamination estimation
    scale_factor = area_gas_acq_diss / _movmean(np.abs(data_gas[:, 0]), 100)[-1]
    contamination_kspace3 = (
        contamination_kspace2 * scale_factor / np.cos(np.pi / 180 * fa_gas)
    )
    # step 4: return subtracted contamination
    return data_dissolved - contamination_kspace3


def calculate_t2star_correction(
    te90: float, t2star_dis: float, field_strength: float
) -> float:
    """Calculate T2* correction factor.

    Args:
        te90 (float): echo time in seconds
        t2star_dis (float): T2* of the dissolved phase in seconds at 3T.
        field_strength (float): B0 field strength in Tesla.
    """
    return np.exp(te90 / (t2star_dis * 3.0 / field_strength)) / np.exp(
        te90 / constants.T2STAR_GAS
    )


def calculate_flipangle_correction(fa_gas: float, fa_dis: float) -> float:
    """Calculate flip angle correction factor.

    Rounds to 3 decimal places.
    Args:
        fa_gas (float): gas flip angle in degrees
        fa_dis (float): dissolved flip angle in degrees
    """
    return np.round((np.sin(fa_gas * np.pi / 180) / np.sin(fa_dis * np.pi / 180)), 3)


def calculate_flipangle_factor(fa_gas: float, fa_dis: float) -> float:
    """Calculate ratio between dissolved and gas flip angles.

    Rounds to 2 decimal places.
    Args:
        fa_gas (float): gas flip angle in degrees
        fa_dis (float): dissolved flip angle in degrees
    """
    return np.round((np.sin(fa_dis * np.pi / 180) / np.sin(fa_gas * np.pi / 180)), 2)


def dixon_decomposition(
    data_dissolved: np.ndarray,
    rbc_m_ratio: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply 1-point dixon decomposition on FID data.

    Applies phase shift to the dissolved data such that the RBC and membrane are
    separated into the imaginary and real channel respectively.
    Does NOT also apply B0 inhomogeneity correction.

    Args:
        data_dissolved (np.ndarray): dissolved FID data of shape
            (n_projections, n_points)
        rbc_m_ratio (float): RBC:m ratio
    Returns:
        Tuple of decomposed RBC and membrane data respectively
    """
    desired_angle = np.arctan2(rbc_m_ratio, 1.0)
    # use k0 to determine the phase shift
    total_dissolved = np.sum(data_dissolved[:, 0])
    current_angle = np.arctan2(np.imag(total_dissolved), np.real(total_dissolved))
    delta_angle = desired_angle - current_angle

    rotated_data = np.multiply(data_dissolved, np.exp(1j * delta_angle))
    return np.imag(rotated_data), np.real(rotated_data)


def smooth(data: np.ndarray, window_size: int = 5) -> np.ndarray:
    """Smooth response data.

    Implements a smoothing function that is equivalent to the MATLAB smooth function.
    Source: https://www.mathworks.com/help/curvefit/smooth.html

    Args:
        data (np.ndarray): 1-D array data to be smoothed.
        window_size (int): size of the smoothing window. Defaults to 5.
    Returns:
        Smoothed data.
    """
    out0 = np.convolve(data, np.ones(window_size, dtype=int), "valid") / window_size
    r = np.arange(1, window_size - 1, 2)
    start = np.cumsum(data[: window_size - 1])[::2] / r
    stop = (np.cumsum(data[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((start, out0, stop))


def bandpass(data: np.ndarray, lowcut: float, highcut: float, fs: float) -> np.ndarray:
    """Bandpass filter.

    Implements a bandpass filter using a butterworth filter.
    Equivalent to MATLAB bandpass filter.

    Args:
        data (np.ndarray): 1-D array data to be filtered.
        lowcut (float): lowcut frequency in Hz.
        highcut (float): highcut frequency in Hz.
        fs (float): sampling frequency.
    Returns:
        Filtered data.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = signal.butter(6, [low, high], analog=False, btype="bandpass", output="sos")
    return np.array(signal.sosfiltfilt(sos, data))


def lowpass(data: np.ndarray, highcut: float, fs: float) -> np.ndarray:
    """Bandpass filter.

    Implements a bandpass filter using a butterworth filter.
    Equivalent to MATLAB bandpass filter.

    Args:
        data (np.ndarray): 1-D array data to be filtered.
        highcut (float): highcut frequency in Hz.
        fs (float): sampling frequency.
    Returns:
        Filtered data.
    """
    nyq = 0.5 * fs
    high = highcut / nyq
    sos = signal.butter(6, high, btype="lowpass", output="sos")
    return np.array(signal.sosfiltfilt(sos, data))

def fit_sine(y: np.ndarray, x: np.ndarray, n: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Fit the data to a sum of n sine waves.

    Args:
        y (np.ndarray): Data to fit. Shape (n,).
        x (np.ndarray): x data. Shape (n,).
        n (int): Number of sine waves to fit to. Defaults to 1.
    Returns:
        Tuple of the fitted data of same shape as input data and the fit parameters.
    """

    def func(x, *args):
        return args[0] * np.sin(args[1] * x + args[2])

    p0 = _sinnstart(x, y, n)
    bounds = _sinbounds(n)
    popt, _ = optimize.curve_fit(
        func,
        x,
        y,
        p0=p0,
        bounds=bounds,
    )
    print("Optimal Parameters:", popt)
    return func(x, *popt), popt


def fit_sine_matlab(y: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fit the data to a sum of n sine waves using matlab engine.

    Args:
        y (np.ndarray): Data to fit. Shape (n,).
        x (np.ndarray): x data. Shape (n,).
    Returns:
        Tuple of the fitted data of same shape as input data and the fit parameters.
    """

    def func(x, *args):
        return args[0] * np.sin(args[1] * x + args[2])

    fit_params = matlab_engine.fit_sine(y, x)
    return func(x, *fit_params), fit_params


# def fit_sine(data: np.ndarray) -> np.ndarray:
#     """Fit the data to a sum of 8 sine waves.

#     Args:
#         data (np.ndarray): 1-D array data to be fitted.
#     Returns:
#         Fitted data. Same shape as input data.
#     """
#     x = np.arange(data.shape[0])
#     y = data

#     def func(x, *args):
#         return (
#             args[0] * np.sin(args[1] * x + args[2])
#             + args[3] * np.sin(args[4] * x + args[5])
#             + args[6] * np.sin(args[7] * x + args[8])
#             + args[9] * np.sin(args[10] * x + args[11])
#             + args[12] * np.sin(args[13] * x + args[14])
#             + args[15] * np.sin(args[16] * x + args[17])
#             + args[18] * np.sin(args[19] * x + args[20])
#             + args[21] * np.sin(args[22] * x + args[23])
#         )

#     p0 = _sinnstart(x, y, 8)
#     bounds = _sinbounds(8)
#     popt, _ = optimize.curve_fit(
#         func,
#         x,
#         y,
#         p0=p0,
#         bounds=bounds,
#     )
#     return func(x, *popt)


def detrend(data: np.ndarray) -> np.ndarray:
    """Remove bi-exponential trend along axis from data.

    Fits the data to a bi-exponential decay function and removes the trend.

    Args:
        data (np.ndarray): 1-D array data to be detrended.
    Returns:
        Detrended data. Same shape as input data.
    """
    x = np.arange(data.shape[0])
    y = data

    def func(x, a, b, c, d):
        return a * np.exp(-b * x) # + c * np.exp(-d * x) ##RH Matching rbc_osc code

    popt, _ = optimize.curve_fit(
        func,
        x,
        y,
        p0=[1, 0.1, 1, 0.1],
        method="trf",
        ftol=1e-6,
        xtol=1e-6,
        max_nfev=1000,
    )
    return (data - func(x, *popt)) / func(x, *popt)


def find_peaks(data: np.ndarray, distance: int = 5) -> np.ndarray:
    """Find peaks in data.

    Implements a peak finding function using scipy.signal.find_peaks.

    Args:
        data (np.ndarray): 1-D array data to be filtered.
        distance (int): minimum distance between peaks. Defaults to 5. Units are
        number of points.

    Returns:
        Array of indices of peaks.
    """
    peaks, _ = signal.find_peaks(data, distance=distance)
    return peaks[np.argwhere(data[peaks] > 0).flatten()]


def get_heartrate(data: np.ndarray, ts: float) -> float:
    """Calculate heart rate from data.

    Implements a heart rate calculation function by finding the strongest peak
    in the fourier domain of the data.

    Args:
        data (np.ndarray): 1-D array data to be filtered.
        ts (float): sampling period in seconds.

    Returns:
        Heart rate in beats per minute.
    """
    fft_data = np.abs(np.fft.fftshift(np.fft.fft(data)))
    freq = np.fft.fftshift(np.fft.fftfreq(len(data), ts))
    return np.abs(freq[np.argmax(fft_data)] * 60)


def awgn(sig: np.ndarray, SNR: float) -> np.ndarray:
    """Add white gaussian noise.

    Args:
        sig (np.ndarray): signal to be added with noise.
        SNR (float): signal to noise ratio in dB.
    """
    sig_power = np.sum(np.abs(sig) ** 2) / len(sig)
    noise_power = sig_power / (10 ** (SNR / 10))

    if np.isreal(sig):
        noise = np.sqrt(noise_power) * np.random.randn(len(sig))
    else:
        noise = np.sqrt(noise_power / 2) * (
            np.random.randn(len(sig)) + 1j * np.random.randn(len(sig))
        )
    return sig + noise


def get_hb_correction(hb: float) -> Tuple[float, float]:
    """Get scaling factors for hb correction.

    Args:
        hb (float): subject hb in g/dL

    Returns:
        rbc_hb_correction_factor (float): rbc hb correction factor
        membrane_hb_correction_factor (float): membrane hb correction factor

    Reference: https://onlinelibrary.wiley.com/doi/10.1002/mrm.29712
    """

    rbc_hb_correction_factor = constants.HbCorrection.R1 + (
        constants.HbCorrection.HB_REF * (1 - constants.HbCorrection.R1) / hb
    )
    membrane_hb_correction_factor = (1 + constants.HbCorrection.M1 * hb) / (
        1
        + constants.HbCorrection.M1 * constants.HbCorrection.HB_REF
        - constants.HbCorrection.M2 * (constants.HbCorrection.HB_REF - hb)
    )

    return rbc_hb_correction_factor, membrane_hb_correction_factor


def sift(raw_fids: np.ndarray) -> np.ndarray:
    """
    Processes raw FIDs using the SIFT method.

    Parameters:
    - raw_fids: ndarray
        The raw FID data of shape (n_points, n_frames).

    Returns:
    - sifted_fids: ndarray
        The processed FID data using the SIFT method of shape
    """

    # Extract all columns except the last
    fids = raw_fids[:, :-1]
    num_points = np.shape(fids)[0]

    # Take Fourier transform of FIDs
    shifted_transform = np.fft.fftshift(np.fft.fft(fids), axes=1)
    thresholded_data = shifted_transform

    # Calculate threshold based on noise for each indirect frequency
    noise = shifted_transform[
        :,
        range(
            int(
                shifted_transform.shape[1]
                - np.round(shifted_transform.shape[1] / 6)
                - 1
            ),
            int(shifted_transform.shape[1]),
        ),
    ]
    threshold = 2 * np.std(abs(noise.T), ddof=1, axis=0) + np.mean(abs(noise.T), axis=0)

    # Zero out all points below the threshold
    for idx in range(0, num_points):
        below_threshold = abs(shifted_transform[idx, :]) < threshold[idx]
        thresholded_data[idx, below_threshold] = 0

    # Remove isolated spikes in indirect frequency spectrum
    flattened_data = thresholded_data.flatten("F")
    non_zero_indices = np.argwhere(abs(flattened_data) > 0.0)
    valid_indices = non_zero_indices[
        np.invert(non_zero_indices > thresholded_data.shape[0])
    ]
    isolated_spikes = flattened_data[valid_indices + 1] == 0
    flattened_data[valid_indices[np.invert(isolated_spikes)]] = 0

    # Take inverse Fourier transform
    reshaped_data = np.reshape(flattened_data, thresholded_data.shape, "F")
    processed_fids = np.fft.ifft(np.fft.ifftshift(reshaped_data, axes=1), axis=1)

    # Replace any sifted FIDs with high residuals
    residuals = np.real(fids - processed_fids)
    high_residuals = (
        np.sum(abs(residuals), axis=0)
        >= (2 * np.std(residuals.flatten("F"))) * residuals.shape[1]
    )
    processed_fids.T[high_residuals] = fids.T[high_residuals]

    # Adjust tail end of FIDs
    processed_fids[num_points - 10 : num_points, :] = processed_fids[
        num_points - 20 : num_points - 10, :
    ]

    # Re-add the last column
    sifted_fids = np.append(processed_fids, np.array([fids[:, -1]]).T, 1)

    return sifted_fids


def moving_mean_2d(x: np.ndarray, window_size: int) -> np.ndarray:
    """
    Computes the moving mean of a 2D matrix along its columns.

    Parameters:
    - x : ndarray
        A 2D matrix for which the moving mean needs to be computed.
    - window_size : int
        Moving

    Returns:
    - result : ndarray
        A 2D matrix containing the moving mean of the input matrix.

    Note:
    The function computes the moving mean with a window size of 5.
    """

    rows, cols = x.shape
    result = np.zeros((rows, cols - window_size + 1), dtype=complex)

    for row in range(rows):
        for col in range(cols - window_size + 1):
            result[row, col] = np.sum(x[row, col : col + window_size]) / window_size

    return result


def get_highpass_filter(
    x: np.ndarray, fc: float = 0.50, fs: float = 50, alpha: float = 2.5
) -> np.ndarray:
    """Calculate the coefficients of a highpass filter.

    Args:
        x (np.ndarray): x data. Shape (n,).
        fc (float): cutoff frequency in Hz. Defaults to 0.50.
        fs (float): sampling frequency in Hz. Defaults to 50.
        alpha (float): alpha parameter for gaussian window. Defaults to 2.5.
    Returns:
        The filter coefficients.
    """
    n = int(2 * np.floor(x.shape[0] / 2 / 3)) - 2
    std = (n - 1) / (2 * alpha)
    return signal.firwin(
        numtaps=n + 1,
        cutoff=fc / (fs / 2),
        window=("gaussian", std),
        pass_zero="highpass",
    )


def get_rbc_norm(y: np.ndarray, t: np.ndarray, t_eval: np.ndarray) -> np.ndarray:
    """Normalize the data to a single exponential decay.

    Args:
        y (np.ndarray): Data to fit. Shape (n,).
        t (np.ndarray): x data. Shape (n,).
        t_eval (np.ndarray): x data to evaluate the fit. Shape (m,).
    Returns:
        The fit data evaluated at t_eval.
    """

    def exp1(t: np.ndarray, a: float, b: float):
        return a * np.exp(-b * t)

    fit_params, _ = optimize.curve_fit(exp1, t, y)
    return exp1(t_eval, *fit_params)


def get_rbc_norm_matlab(y: np.ndarray, t: np.ndarray, t_eval: np.ndarray) -> np.ndarray:
    """Normalize the data to a single exponential decay fitted using the matlab engine.

    Args:
        y (np.ndarray): Data to fit. Shape (n,).
        t (np.ndarray): x data. Shape (n,).
        t_eval (np.ndarray): x data to evaluate the fit. Shape (m,).
    Returns:
        The fit data evaluated at t_eval.
    """

    def exp1(t: np.ndarray, a: float, b: float):
        return a * np.exp(b * t)

    fit_params = matlab_engine.fit_exp1(y, t)
    return exp1(t_eval, *fit_params)


# def fit_sine(y: np.ndarray, x: np.ndarray, n: int = 1) -> Tuple[np.ndarray, np.ndarray]:
#     """Fit the data to a sum of n sine waves.

#     Args:
#         y (np.ndarray): Data to fit. Shape (n,).
#         x (np.ndarray): x data. Shape (n,).
#         n (int): Number of sine waves to fit to. Defaults to 1.
#     Returns:
#         Tuple of the fitted data of same shape as input data and the fit parameters.
#     """

#     def func(x, *args):
#         return args[0] * np.sin(args[1] * x + args[2])

#     p0 = _sinnstart(x, y, n)
#     bounds = _sinbounds(n)
#     popt, _ = optimize.curve_fit(
#         func,
#         x,
#         y,
#         p0=p0,
#         bounds=bounds,
#     )
#     print("Optimal Parameters:", popt)
#     return func(x, *popt), popt

# def detrend(data: np.ndarray) -> np.ndarray:
#     """Remove bi-exponential trend along axis from data.

#     Fits the data to a bi-exponential decay function and removes the trend.

#     Args:
#         data (np.ndarray): 1-D array data to be detrended.
#     Returns:
#         Detrended data. Same shape as input data.
#     """
#     x = np.arange(data.shape[0])
#     y = data

#     def func(x, a, b, c, d):
#         return a * np.exp(-b * x)  # + c * np.exp(-d * x)

#     popt, _ = optimize.curve_fit(
#         func,
#         x,
#         y,
#         p0=[1, 0.1, 1, 0.1],
#         method="trf",
#         ftol=1e-6,
#         xtol=1e-6,
#         max_nfev=600,
#     )
#     return (data - func(x, *popt)) / func(x, *popt)


# def find_peaks(data: np.ndarray, distance: int = 5) -> np.ndarray:
#     """Find peaks in data.

#     Implements a peak finding function using scipy.signal.find_peaks.

#     Args:
#         data (np.ndarray): 1-D array data to be filtered.
#         distance (int): minimum distance between peaks. Defaults to 5. Units are
#         number of points.

#     Returns:
#         Array of indices of peaks.
#     """
#     peaks, _ = signal.find_peaks(data, distance=distance)
#     return peaks[np.argwhere(data[peaks] > 0).flatten()]


# def get_heartrate(data: np.ndarray, ts: float) -> float:
#     """Calculate heart rate from data.

#     Implements a heart rate calculation function by finding the strongest peak
#     in the fourier domain of the data.

#     Args:
#         data (np.ndarray): 1-D array data to be filtered.
#         ts (float): sampling period in seconds.

#     Returns:
#         Heart rate in beats per minute.
#     """
#     fft_data = np.abs(np.fft.fftshift(np.fft.fft(data)))
#     freq = np.fft.fftshift(np.fft.fftfreq(len(data), ts))
#     # Exclude the DC frequency by considering only non-DC frequencies
#     non_dc_indices = np.nonzero(freq)
#     fft_data_non_dc = fft_data[non_dc_indices]
#     freq_non_dc = freq[non_dc_indices]

#     return np.abs(freq_non_dc[np.argmax(fft_data_non_dc)] * 60)


# def awgn(sig: np.ndarray, SNR: float) -> np.ndarray:
#     """Add white gaussian noise.

#     Args:
#         sig (np.ndarray): signal to be added with noise.
#         SNR (float): signal to noise ratio in dB.
#     """
#     sig_power = np.sum(np.abs(sig) ** 2) / len(sig)
#     noise_power = sig_power / (10 ** (SNR / 10))

#     if np.isreal(sig):
#         noise = np.sqrt(noise_power) * np.random.randn(len(sig))
#     else:
#         noise = np.sqrt(noise_power / 2) * (
#             np.random.randn(len(sig)) + 1j * np.random.randn(len(sig))
#         )
#     return sig + noise


def find_high_low_indices(
    data: np.ndarray,
    peak_distance: int,
    distance_threshold: float = 0.2,
    same_length: bool = True,
    method: str = constants.BinningMethods.PEAKS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Find indices of high and low signal bins.

    Args:
        data (np.ndarray): RBC 1-D data of shape (n_projections,)
        peak_distance (int): distance between peaks in number of points.
        distance_threshold (float): threshold for neighbouring peaks. Defaults to 0.2.
            Value must be between 0 and 1 with 0 being taking only the found peaks and 1
            being taking all points between the peaks.
        same_length (bool): whether to force high and low bins are of the same length.

    Returns:
        Tuple of indices of high and low signal bins respectively.
    """
    high_indices = np.array([])
    low_indices = np.array([])

    if method == constants.BinningMethods.PEAKS:
        high_peaks = find_peaks(data=data, distance=int(0.6 * peak_distance))
        low_peaks = find_peaks(data=-data, distance=int(0.6 * peak_distance))

        left = np.ceil(peak_distance * distance_threshold / 2).astype(int)
        right = left + 1
        for peak in high_peaks:
            high_indices = np.append(high_indices, np.arange(peak - left, peak + right))
        for peak in low_peaks:
            low_indices = np.append(low_indices, np.arange(peak - left, peak + right))
    elif method == constants.BinningMethods.THRESHOLD:
        data_norm = (data - np.mean(data)) / np.std(data)
        high_indices = np.argwhere(data_norm > 0.7).flatten()
        low_indices = np.argwhere(data_norm < -0.7).flatten()
    else:
        raise ValueError(f"Method {method} not implemented.")

    # remove indices that are below zero and above length of the data
    high_indices = np.delete(high_indices, np.argwhere(high_indices < 0))
    low_indices = np.delete(low_indices, np.argwhere(low_indices < 0))
    high_indices = np.delete(high_indices, np.argwhere(high_indices >= len(data)))
    low_indices = np.delete(low_indices, np.argwhere(low_indices >= len(data)))
    if same_length:
        if len(high_indices) > len(low_indices):
            high_indices = high_indices[: len(low_indices)]
        elif len(low_indices) > len(high_indices):
            low_indices = low_indices[: len(high_indices)]
    return np.sort(high_indices).astype(int), np.sort(low_indices).astype(int)


def find_indices_sliding_window(
    data: np.ndarray,
    peak_distance: int,
    distance_threshold: float = 0.2,
    window_spacing: int = 5,
) -> list[np.ndarray]:
    """Find indices of all sliding windows from the high peaks.

    Args:
        data (np.ndarray): RBC 1-D data of shape (n_projections,)
        peak_distance (int): distance between peaks in number of points.
        distance_threshold (float): threshold for neighbouring peaks. Defaults to 0.2.
            Value must be between 0 and 1 with 0 being taking only the found peaks and 1
            being taking all points between the peaks.
        window_spacing (int): spacing between the windows.
    Returns:
        List of list of indices respectively.
    """
    high_indices = np.array([])

    high_peaks = find_peaks(data=data, distance=int(0.6 * peak_distance))

    left = np.ceil(peak_distance * distance_threshold / 2).astype(int)
    right = left + 1
    for peak in high_peaks:
        high_indices = np.append(high_indices, np.arange(peak - left, peak + right))
    n_windows = int(np.mean(np.diff(high_peaks))) // window_spacing
    out_indices = []

    for i in range(n_windows):
        cur_indices = high_indices + window_spacing * i
        # remove indices that go are below zero and above length of the data
        cur_indices = np.delete(cur_indices, np.argwhere(cur_indices < 0))
        cur_indices = np.delete(cur_indices, np.argwhere(cur_indices >= len(data)))
        out_indices.append(cur_indices.astype(int))

    return out_indices


def find_indices_sliding_window_start(
    data: np.ndarray,
    window_size: int,
    window_spacing: int,
) -> list[np.ndarray]:
    """Find indices of all sliding windows starting from the beginning.

    Ensures that all the windows have the same number of projections.
    Args:
        data (np.ndarray): RBC 1-D data of shape (n_projections,)
        window_size (int): Size of the moving window.
        window_spacing (int): Spacing between the windows.

    Returns:
        List of indices arrays for each window.
    """
    out_indices = []
    n_windows = int(np.floor((data.shape[0] - window_size) / window_spacing))
    for i in range(n_windows):
        cur_indices = np.arange(i * window_spacing, i * window_spacing + window_size)
        out_indices.append(cur_indices.astype(int))
    return out_indices


def moving_average_filter(data: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Apply a moving average filter to 1D data.

    Args:
        data (np.ndarray): 1D array of data.
        window_size (int): Size of the moving window for averaging.

    Returns:
        np.ndarray: Filtered data after applying the moving average.

    Raises:
        ValueError: If the window size is not a positive odd integer.

    """
    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError("Window size must be a positive odd integer.")

    half_window = window_size // 2
    filtered_data = np.convolve(data, np.ones(window_size) / window_size, mode="same")
    return filtered_data[half_window:-half_window]


def median_filter(data: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Apply a median filter to 1D data.

    Args:
        data (np.ndarray): 1D array of data.
        window_size (int): Size of the moving window for median filtering.

    Returns:
        np.ndarray: Filtered data after applying the median filter.

    Raises:
        ValueError: If the window size is not a positive odd integer.

    """
    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError("Window size must be a positive odd integer.")

    half_window = window_size // 2
    filtered_data = np.zeros_like(data)
    for i in range(half_window, len(data) - half_window):
        window = data[i - half_window : i + half_window + 1]
        filtered_data[i] = np.median(window)

    return filtered_data


def wavelet_denoise(
    signal_: np.ndarray, wavelet: str = "db4", level: int = 1
) -> np.ndarray:
    """
    Apply wavelet denoising to a 1D signal.

    Args:
        signal (np.ndarray): Input signal.
        wavelet (str): Name of the wavelet function to use. Defaults to 'db4'.
        level (int): Decomposition level for the wavelet transform. Defaults to 1.

    Returns:
        np.ndarray: Denoised signal.

    """
    # Perform wavelet decomposition
    coeffs = pywt.wavedec(signal_, wavelet, level=level)

    # Estimate the noise level based on the standard deviation of the highest-frequency coefficients
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745

    # Apply soft thresholding to the detail coefficients
    denoised_coeffs = [pywt.threshold(c, value=sigma, mode="soft") for c in coeffs]

    # Reconstruct the denoised signal
    denoised_signal = pywt.waverec(denoised_coeffs, wavelet)

    return denoised_signal


def get_vol_correction(vol: float, expected_lung_volume: float) -> Tuple[float, float, float]:
    """Get scaling factors for volume correction.
    Args:
        vol (float): volume of lung mask in L
        expected_lung_volume (float): user input target lung volume in L
    Returns:
        vol_correction_factor_rbc (float): rbc volume correction factor
        vol_correction_factor_membrane (float): membrane volume correction factor
    """
    vol2 = expected_lung_volume

    vol_correction_factor_rbc = (vol *( 1 + constants.VolCorrection.ALPHA_RBC) + vol2 * (
        1 - constants.VolCorrection.ALPHA_RBC))/(vol * (1 - constants.VolCorrection.ALPHA_RBC
                                                ) + vol2 * (1 + constants.VolCorrection.ALPHA_RBC))
    print("VCF_RBC = " + str(vol_correction_factor_rbc))
    vol_correction_factor_membrane = (vol *( 1 + constants.VolCorrection.ALPHA_MEM) + vol2 * (
        1 - constants.VolCorrection.ALPHA_MEM))/(vol * (1 - constants.VolCorrection.ALPHA_MEM
                                                ) + vol2 * (1 + constants.VolCorrection.ALPHA_MEM))
    print("VCF_mem = " + str(vol_correction_factor_membrane))

    return vol_correction_factor_rbc, vol_correction_factor_membrane, vol2


# def boxcox(data: np.ndarray) -> tuple[np.ndarray, float]:
#     """Apply box cox transformation on data.

#     Args:
#         data (np.ndarray): data to be transformed of shape (n,)
#     Returns:
#         Tuple of transformed data and box cox lambda
#     """
#     return stats.boxcox(data)


# def inverse_boxcox(
#     boxcox_lambda: float, data: np.ndarray, scale_factor: float
# ) -> np.ndarray:
#     """Apply inverse box cox transformation on data.

#     Args:
#         boxcox_lambda (float): box cox lambda
#         data (np.ndarray): data to be transformed of shape (n,)
#         scale_factor (float): scale factor to be applied to the data
#     """
#     return np.power(boxcox_lambda * data + 1, 1 / boxcox_lambda) - scale_factor


# def remove_gasphase_contamination(
#     data_dissolved: np.ndarray,
#     data_gas: np.ndarray,
#     dwell_time: float,
#     freq_gas_acq_diss: float,
#     phase_gas_acq_diss: float,
#     area_gas_acq_diss: float,
#     fa_gas: float,
# ) -> np.ndarray:
#     """Remove gas phase contamination in dissolved k-space.

#     Takes gas phase k-space and modifies it using NMR fits and gas phase k0
#     to produce the expected gas phase contamination k-space data which is
#     then removed from the initial contaminated dissolved phase k-space.

#     Args:
#         data_dissolved (np.ndarray): dissolved k-space data of shape
#             (n_projections, n_points)
#         data_gas (np.ndarray): gas phase k-space data of shape
#             (n_projections, n_points)
#         dwell_time (float): dwell time in seconds.
#         freq_gas_acq_diss (float): gas frequency offset in dissolved
#             spectra acquisition in Hz.
#         phase_gas_acq_diss (float): gas phase in dissolved spectra acquisition.
#             in degrees.
#         area_gas_acq_diss (float): gas area in dissolved spectra acquisition.
#         fa_gas (float): gas flip angle in degrees.
#     Returns:
#         Gas phase corrected dissolved k-space data of shape (n_projections, n_points)
#     Author: Matt Willmering
#     Paper: https://pubmed.ncbi.nlm.nih.gov/33665905/
#     """
#     # step 0: calculate parameters
#     arr_t = dwell_time * np.arange(data_dissolved.shape[1])
#     # step 1: modulate contamination (gas) to dissolved frequency - first order
#     # phase approximation
#     phase_shift1 = 2 * np.pi * freq_gas_acq_diss * arr_t  # calculate phase accumulation
#     contamination_kspace1 = data_gas * np.exp(1j * phase_shift1)
#     # step 2: zero order phase shift of contamination estimation
#     phase_shift2 = phase_gas_acq_diss - 180 / np.pi * np.mean(np.angle(data_gas[:, 0]))
#     contamination_kspace2 = contamination_kspace1 * np.exp(
#         1j * np.pi / 180 * phase_shift2
#     )
#     # step 3: scale contamination estimation
#     scale_factor = area_gas_acq_diss / _movmean(np.abs(data_gas[:, 0]), 100)[-1]
#     contamination_kspace3 = (
#         contamination_kspace2 * scale_factor / np.cos(np.pi / 180 * fa_gas)
#     )
#     # step 4: return subtracted contamination
#     return data_dissolved - contamination_kspace3


# def dixon_decomposition(
#     data_dissolved: np.ndarray,
#     rbc_m_ratio: float,
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """Apply 1-point dixon decomposition on FID data.

#     Applies phase shift to the dissolved data such that the RBC and membrane are
#     separated into the imaginary and real channel respectively.
#     Does NOT also apply B0 inhomogeneity correction.

#     Args:
#         data_dissolved (np.ndarray): dissolved FID data of shape
#             (n_projections, n_points)
#         rbc_m_ratio (float): RBC:m ratio
#     Returns:
#         Tuple of decomposed RBC and membrane data respectively
#     """
#     desired_angle = np.arctan2(rbc_m_ratio, 1.0)
#     # use k0 to determine the phase shift
#     total_dissolved = np.sum(data_dissolved[:, 0])
#     current_angle = np.arctan2(np.imag(total_dissolved), np.real(total_dissolved))
#     delta_angle = desired_angle - current_angle

#     rotated_data = np.multiply(data_dissolved, np.exp(1j * delta_angle))
#     return np.imag(rotated_data), np.real(rotated_data)


# def smooth(data: np.ndarray, window_size: int = 5) -> np.ndarray:
#     """Smooth response data.

#     Implements a smoothing function that is equivalent to the MATLAB smooth function.
#     Source: https://www.mathworks.com/help/curvefit/smooth.html

#     Args:
#         data (np.ndarray): 1-D array data to be smoothed.
#         window_size (int): size of the smoothing window. Defaults to 5.
#     Returns:
#         Smoothed data.
#     """
#     out0 = np.convolve(data, np.ones(window_size, dtype=int), "valid") / window_size
#     r = np.arange(1, window_size - 1, 2)
#     start = np.cumsum(data[: window_size - 1])[::2] / r
#     stop = (np.cumsum(data[:-window_size:-1])[::2] / r)[::-1]
#     return np.concatenate((start, out0, stop))


# def bandpass(data: np.ndarray, lowcut: float, highcut: float, fs: float) -> np.ndarray:
#     """Bandpass filter.

#     Implements a bandpass filter using a butterworth filter.
#     Equivalent to MATLAB bandpass filter.

#     Args:
#         data (np.ndarray): 1-D array data to be filtered.
#         lowcut (float): lowcut frequency in Hz.
#         highcut (float): highcut frequency in Hz.
#         fs (float): sampling frequency.
#     Returns:
#         Filtered data.
#     """
#     nyq = 0.5 * fs
#     low = lowcut / nyq
#     high = highcut / nyq
#     sos = signal.butter(6, [low, high], analog=False, btype="bandpass", output="sos")
#     return np.array(signal.sosfiltfilt(sos, data))


# def lowpass(data: np.ndarray, highcut: float, fs: float) -> np.ndarray:
#     """Bandpass filter.

#     Implements a bandpass filter using a butterworth filter.
#     Equivalent to MATLAB bandpass filter.

#     Args:
#         data (np.ndarray): 1-D array data to be filtered.
#         highcut (float): highcut frequency in Hz.
#         fs (float): sampling frequency.
#     Returns:
#         Filtered data.
#     """
#     nyq = 0.5 * fs
#     high = highcut / nyq
#     sos = signal.butter(6, high, btype="lowpass", output="sos")
#     return np.array(signal.sosfiltfilt(sos, data))


def find_npeaks(data: np.ndarray, npeaks: Optional[int] = None) -> np.ndarray:
    """Find peaks in data.

    Implements a peak finding function using scipy.signal.find_peaks.

    Args:
        data (np.ndarray): 1-D array data to be filtered.
        npeaks (Optional[int]): number of peaks to return in descending order.

    Returns:
        Array of indices of peaks.
    """
    peaks, _ = signal.find_peaks(data)
    if npeaks is not None:
        peaks = peaks[np.argsort(data[peaks])][::-1][: min(npeaks, len(peaks))]
    return peaks


# def get_hb_correction(hb: float) -> Tuple[float, float]:
#     """Get scaling factors for hb correction.

#     Args:
#         hb (float): subject hb in g/dL

#     Returns:
#         rbc_hb_correction_factor (float): rbc hb correction factor
#         membrane_hb_correction_factor (float): membrane hb correction factor

#     Reference: https://onlinelibrary.wiley.com/doi/10.1002/mrm.29712
#     """

#     rbc_hb_correction_factor = constants.HbCorrection.R1 + (
#         constants.HbCorrection.HB_REF * (1 - constants.HbCorrection.R1) / hb
#     )
#     membrane_hb_correction_factor = (1 + constants.HbCorrection.M1 * hb) / (
#         1
#         + constants.HbCorrection.M1 * constants.HbCorrection.HB_REF
#         - constants.HbCorrection.M2 * (constants.HbCorrection.HB_REF - hb)
#     )

#     return rbc_hb_correction_factor, membrane_hb_correction_factor


# def calculate_t2star_correction(te90: float) -> float:
#     """Calculate T2* correction factor.

#     Rounds to 3 decimal places.
#     Args:
#         te90 (float): echo time in seconds
#     """
#     return np.round(np.exp(te90 / 2e-3) / np.exp(te90 / 5e-2), 3)


# def calculate_flipangle_correction(fa_gas: float, fa_dis: float) -> float:
#     """Calculate flip angle correction factor.

#     Rounds to 3 decimal places.
#     Args:
#         fa_gas (float): gas flip angle in degrees
#         fa_dis (float): dissolved flip angle in degrees
#     """
#     return np.round(
#         (100 * np.sin(fa_gas * np.pi / 180) / np.sin(fa_dis * np.pi / 180)), 3
#     )


# def calculate_flipangle_factor(fa_gas: float, fa_dis: float) -> float:
#     """Calculate ratio between dissolved and gas flip angles.

#     Rounds to 2 decimal places.
#     Args:
#         fa_gas (float): gas flip angle in degrees
#         fa_dis (float): dissolved flip angle in degrees
#     """
#     return np.round((np.sin(fa_dis * np.pi / 180) / np.sin(fa_gas * np.pi / 180)), 2)


def calculate_decay_factor(data: np.ndarray, t2star, dwell_time: float) -> np.ndarray:
    """Calculate decay factor.

    Rounds to 3 decimal places.
    Args:
        data (np.ndarray): data to be corrected of shape (n_proj, n_points)
        t2star (float): T2* in seconds
        dwell_time (float): dwell time in seconds
    """
    k0 = np.abs(data[:, 0])
    k = np.zeros(data.shape, dtype=complex)
    relaxation = np.zeros((data.shape[1],))
    for i in range(data.shape[1]):
        relaxation[i] = np.exp(-dwell_time * i / t2star)
    for i in range(data.shape[0]):
        k[i, :] = k0[i] * relaxation
    return k
