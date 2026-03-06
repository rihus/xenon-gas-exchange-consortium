"""Spectroscopy util functions."""
import math
import sys

from typing import Any, Optional, Tuple
import numpy as np

import matlab_engine
import spect.nmr_timefit as fit
from utils import constants, metrics, signal_utils
sys.path.append("..")

def get_breathhold_indices(
    t: np.ndarray, start_time: int, end_time: int
) -> Tuple[int, int]:
    """Get the start and stop index based on the start and stop time.

    Find the index in the time array corresponding to the start time and the end time.
    If the start index is not found, return 0.
    If the stop index is not found return the last index of the array.

    Args:
        t (np.ndarray): array of time points each FID is collected in units of seconds.
        start_time (int): start time (in seconds) of window to analyze t.
        end_time (int): stop time (in seconds) of window to analyze t.

    Returns:
        Tuple of the indices corresponding to the start time and stop time.
    """

    def round_up(x: float, decimals: int = 0) -> float:
        """Round number to the nearest decimal place.

        Args:
            x: floating point number to be rounded up.
            decimals: number of decimal places to round by.

        Returns:
            rounded up value of x.
        """
        return math.ceil(x * 10**decimals) / 10**decimals

    start_ind = np.argwhere(np.array([round_up(x, 2) for x in t]) == start_time)
    end_ind = np.argwhere(np.array([round_up(x, 2) for x in t]) == end_time)

    if np.size(start_ind) == 0:
        start_ind = [0]
    if np.size(end_ind) == 0:
        end_ind = [np.size(t)]
    return (
        int(start_ind[int(np.floor(np.size(start_ind) / 2))]),
        int(end_ind[int(np.floor(np.size(end_ind) / 2))]),
    )

def get_frequency_guess(
    data: Optional[np.ndarray], center_freq: float, rf_excitation: int
):
    """Get the three-peak initial frequency guess.

    This can be modified in the future to include automated peak finding.

    Args:
        data (np.ndarray): FID data of shape (n_points, 1) or (n_points, ).
        center_freq (float): center frequency in MHz.
        rf_excitation (int): excitation frequency in ppm.

    Returns: 3-element array of initial frequency guesses corresponding to the RBC,
        membrane, and gas frequencys in MHz
    """
    if rf_excitation == 208:
        return np.array([10, -21.7, -208.4]) * center_freq
    if rf_excitation == 218:
        return np.array([0, -21.7, -218.0]) * center_freq
    #
    raise ValueError("Invalid excitation frequency {}".format(rf_excitation))


# def _get_frequency_guess(
#     data: Optional[np.ndarray], center_freq: float, rf_excitation: int
# ) -> np.ndarray:
#     """Get the three-peak initial frequency guess.

#     This can be modified in the future to include automated peak finding.

#     Args:
#         data (np.ndarray): FID data of shape (n_points, 1) or (n_points, ).
#         center_freq (float): center frequency in MHz.
#         rf_excitation (int): excitation frequency in ppm.

#     Returns: 3-element array of initial frequency guesses corresponding to the RBC,
#         membrane, and gas frequencys in MHz
#     """
#     rbc_freq = 217.2 - rf_excitation
#     membrane_freq = 197.7 - rf_excitation
#     gas_freq = 0 - rf_excitation

#     return np.array([rbc_freq, membrane_freq, gas_freq]) * center_freq


def get_area_guess(data: Optional[np.ndarray], center_freq: float, rf_excitation: int):
    """Get the three-peak initial area guess.

    This can be modified in the future to include automated peak finding.

    Args:
        data (np.ndarray): FID data of shape (n_points, 1) or (n_points, ).
        center_freq (float): center frequency in MHz.
        rf_excitation (int): excitation frequency in ppm.

    Returns: 3-element array of initial area guesses corresponding to the RBC,
        membrane, and gas frequencys in MHz
    """
    if rf_excitation == 208:
        return np.array([1, 1, 1])
    if rf_excitation == 218:
        return np.array([1, 1, 1])
    #
    raise ValueError("Invalid excitation frequency {}".format(rf_excitation))


def _get_positive_phase(phase: np.ndarray) -> np.ndarray:
    """Get the positive phase, in reference to membrane phase.

    Args:
        phase (np.ndarray): phase array in degrees, in the order of RBC, membrane, gas.

    Returns:
        phase (np.ndarray): phase in degrees between 0 and 360.
    """
    phase = phase - phase[1]
    phase[phase < 0] = phase[phase < 0] + 360
    return phase


# def get_breathhold_indices(
#     t: np.ndarray, start_time: float = 2, end_time: float = 7
# ) -> Tuple[int, int]:
#     """Get the start and stop index based on the start and stop time.

#     Find the index in the time array corresponding to the start time and the end time.
#     If the start index is not found, return 0.
#     If the stop index is not found return the last index of the array.

#     Args:
#         t (np.ndarray): array of time points each FID is collected in units of seconds.
#         start_time (int): start time (in seconds) of window to analyze t.
#         end_time (int): stop time (in seconds) of window to analyze t.

#     Returns:
#         Tuple of the indices corresponding to the start time and stop time.
#     """

#     start_ind = np.argwhere(np.array([np.round(x, 1) for x in t]) == start_time)
#     end_ind = np.argwhere(np.array([np.round(x, 1) for x in t]) == end_time)

#     if np.size(start_ind) == 0:
#         start_ind = [0]
#     if np.size(end_ind) == 0:
#         end_ind = [np.size(t)]

#     return (
#         int(start_ind[int(np.floor(np.size(start_ind) / 2))]),
#         int(end_ind[int(np.floor(np.size(end_ind) / 2))]),
#     )


def calculate_static_spectroscopy(
    fid: np.ndarray,
    sample_time: float = 1.95e-05,
    tr: float = 0.015,
    center_freq: float = 34.09,
    rf_excitation: int = 218,
    n_avg: Optional[int] = None,
    n_avg_seconds: int = 1,
    method: str = "voigt",
    plot: bool = False,
) -> Tuple[float, Any]:
    """Fit static spectroscopy data to Voigt model and extract RBC:M ratio.

    The RBC:M ratio is defined as the ratio of the fitted RBC peak area to the membrane
    peak area.
    Args:
        fid (np.ndarray): Dissolved phase FIDs in format (n_points, n_frames).
        sample_time (float): Dwell time in seconds.
        tr (float): TR in seconds.
        center_freq (float): Center frequency in MHz.
        rf_excitation (int, optional): _description_. Excitation frequency in ppm.
        n_avg (int, optional): Number of FIDs to average for static spectroscopy.
        n_avg_seconds (int): Number of seconds to average for
            static spectroscopy.
        plot (bool, optional): Plot the fit. Defaults to False.

    Returns:
        Tuple of RBC:M ratio and fit object
    """
    t = np.array(range(0, np.shape(fid)[0])) * sample_time
    t_tr = np.array(range(0, np.shape(fid)[1])) * tr

    start_ind, _ = get_breathhold_indices(t=t_tr, start_time=2, end_time=10)
    # calculate number of FIDs to average
    if n_avg:
        n_avg = n_avg
    else:
        n_avg = int(n_avg_seconds / tr)

    end_ind = np.min([len(fid[0, :]) - 1, start_ind + n_avg + 1])
    data_dis_avg = np.average(fid[:, start_ind:end_ind], axis=1)
    fit_obj = fit.NMR_TimeFit(
        ydata=data_dis_avg,
        tdata=t,
        area=get_area_guess(
            data=None, center_freq=center_freq, rf_excitation=rf_excitation
        ),
        freq=get_frequency_guess(
            data=None, center_freq=center_freq, rf_excitation=rf_excitation
        ),
        fwhmL=np.array([8.8, 5.0, 2.0]) * center_freq,
        fwhmG=np.array([0, 6.1, 0]) * center_freq,
        phase=np.array([0, 0, 0]),
        line_broadening=0,
        zeropad_size=np.size(t),
        method=method,
    )
    lb = np.stack(
        (
            [-np.inf, -np.inf, -np.inf],
            [-np.inf, -np.inf, -np.inf],
            [-np.inf, -np.inf, -np.inf],
            [-np.inf, -np.inf, -np.inf],
            [-np.inf, -np.inf, -np.inf],
        )
    ).flatten()
    ub = np.stack(
        (
            [+np.inf, +np.inf, +np.inf],
            [+np.inf, +np.inf, +np.inf],
            [+np.inf, +np.inf, +np.inf],
            [+np.inf, +np.inf, +np.inf],
            [+np.inf, +np.inf, +np.inf],
        )
    ).flatten()
    bounds = (lb, ub)
    fit_obj.fit_time_signal_residual(bounds=bounds)
    if plot:
        fit_obj.plot_time_spect_fit()
    rbc_m_ratio = fit_obj.area[0] / np.sum(fit_obj.area[1])
    return rbc_m_ratio, fit_obj


def fit_static_spectroscopy(
    fids: np.ndarray,
    dwell_time: float = 1.95e-05,
    tr: float = 0.015,
    center_freq: float = 34.09,
    rf_excitation: int = 218,
    n_avg: Optional[int] = None,
    n_avg_seconds: int = 1,
    method: str = "voigt",
    average_all: bool = True,
) -> Tuple[float, Any]:
    """Fit static spectroscopy data to Voigt model and extract RBC:M ratio.

    The RBC:M ratio is defined as the ratio of the fitted RBC peak area to the membrane
    peak area.
    Args:
        fid (np.ndarray): Dissolved phase FIDs in format (n_points, n_frames).
        dwell_time (float): Dwell time in seconds.
        tr (float): TR in seconds.
        center_freq (float): Center frequency in MHz.
        rf_excitation (int, optional): _description_. Excitation frequency in ppm.
        n_avg (int, optional): Number of FIDs to average for static spectroscopy.
        n_avg_seconds (int): Number of seconds to average for
            static spectroscopy.

    Returns:
        Tuple of RBC:M ratio and fit object
    """
    t = np.array(range(0, np.shape(fids)[0])) * dwell_time
    t_tr = np.array(range(1, np.shape(fids)[1] + 1)) * tr

    start_ind, _ = get_breathhold_indices(t=t_tr, start_time=2, end_time=10)
    # calculate number of FIDs to average
    if n_avg:
        n_avg = n_avg
    else:
        n_avg = int(np.ceil(n_avg_seconds / tr))

    end_ind = np.min([len(fids[0, :]) - 1, start_ind + n_avg + 1])
    if average_all:
        start_ind = 0
        end_ind = len(fids[0, :])
    data_dis_avg = np.mean(fids[:, start_ind:end_ind], axis=1)
    area = (
        get_area_guess(
            data=None, center_freq=center_freq, rf_excitation=rf_excitation
        ),
    )
    freq = (
        get_frequency_guess(
            data=None, center_freq=center_freq, rf_excitation=rf_excitation
        ),
    )
    fwhmL = (np.array([8.8, 5.0, 1.2]) * center_freq,)
    fwhmG = (np.array([0, 6.1, 0]) * center_freq,)
    phase = (np.array([0, 0, 0]),)
    fit_params0 = np.array([area, freq, fwhmL, fwhmG, phase]).flatten()
    # fit the data
    fit_params = matlab_engine.timefit_lsqcurvefit(
        fit_params0, t, fids[:, start_ind:end_ind]
    )

    # define the fit object
    fit_obj = fit.NMR_TimeFit(
        ydata=data_dis_avg,
        tdata=t,
        area=fit_params[0:3],
        freq=fit_params[3:6],
        fwhmL=fit_params[6:9],
        fwhmG=fit_params[9:12],
        phase=fit_params[12:15],
        line_broadening=0,
        zeropad_size=np.size(t),
        method=method,
    )
    rbc_m_ratio = fit_obj.area[0] / fit_obj.area[1]

    # prepare the output dictionary
    out_dict = {
        constants.SpectIOFields.DATA_DIS_AVG: data_dis_avg,
        constants.SpectIOFields.T_SPECTRA: t,
        constants.SpectIOFields.FIT_PARAMS: fit_params,
        constants.SpectIOFields.MEMBRANE_AREA: (fit_obj.area[1] / fit_obj.area[1]),
        constants.SpectIOFields.MEMBRANE_FWHM: fit_obj.fwhmL[1] / center_freq,
        constants.SpectIOFields.MEMBRANE_FWHMG: fit_obj.fwhmG[1] / center_freq,
        constants.SpectIOFields.MEMBRANE_PHASE: _get_positive_phase(fit_obj.phase)[1],
        constants.SpectIOFields.MEMBRANE_SHIFT_PPM: (fit_obj.freq[1] - fit_obj.freq[-1])
        / center_freq,
        constants.SpectIOFields.MEMBRANE_SNR_TAIL: metrics.get_snr_tail(
            fit_obj.area, fit_obj.ydata, fit_obj.get_time_function(fit_obj.tdata)
        )[1],
        constants.SpectIOFields.RBC_AREA: fit_obj.area[0] / fit_obj.area[1],
        constants.SpectIOFields.RBC_FWHM: fit_obj.fwhmL[0] / center_freq,
        constants.SpectIOFields.RBC_PHASE: _get_positive_phase(fit_obj.phase)[0],
        constants.SpectIOFields.RBC_SHIFT_PPM: (fit_obj.freq[0] - fit_obj.freq[-1])
        / center_freq,
        constants.SpectIOFields.RBC_SNR_TAIL: metrics.get_snr_tail(
            fit_obj.area, fit_obj.ydata, fit_obj.get_time_function(fit_obj.tdata)
        )[0],
    }

    return rbc_m_ratio, out_dict


def fit_dynamic_spectroscopy(
    fids: np.ndarray,
    dwell_time: float = 1.95e-05,
    tr: float = 0.015,
    center_freq: float = 34.09,
    rf_excitation: int = 218,
    method: str = "voigt",
    window_size: int = 5,
) -> dict[str, Any]:
    """Fit dynamic spectroscopy data to Voigt model.

    The RBC:M ratio is defined as the ratio of the fitted RBC peak area to the membrane
    peak area.
    Args:
        fid (np.ndarray): Dissolved phase FIDs in format (n_points, n_frames).
        dwell_time (float): Dwell time in seconds.
        tr (float): TR in seconds.
        center_freq (float): Center frequency in MHz.
        rf_excitation (int, optional): _description_. Excitation frequency in ppm.
        n_avg (int): Number of FIDs to average for dynamic spectroscopy.

    Returns:
        Tuple of RBC:M ratio and fit object
    """
    t = np.array(range(0, np.shape(fids)[0])) * dwell_time
    t_tr = np.array(range(1, np.shape(fids)[1] + 1)) * tr
    # SIFT raw fids
    fids = matlab_engine.sift(fids)
    # remove the last 2 frames
    # TODO: I am not sure why we're doing this, but copying the code.
    fids = fids[:, :-2]
    # average over a window to get starting guess
    area = (
        get_area_guess(
            data=None, center_freq=center_freq, rf_excitation=rf_excitation
        ),
    )
    freq = (
        get_frequency_guess(
            data=None, center_freq=center_freq, rf_excitation=rf_excitation
        ),
    )
    fwhmL = (np.array([8.8, 5.0, 1.2]) * center_freq,)
    fwhmG = (np.array([0, 6.1, 0]) * center_freq,)
    phase = (np.array([0, 0, 0]),)
    fit_params0 = np.array([area, freq, fwhmL, fwhmG, phase]).flatten()
    fit_params_guess = matlab_engine.timefit_lsqcurvefit(fit_params0, t, fids)

    fit_guess_obj = fit.NMR_TimeFit(
        ydata=np.mean(fids[:, 99:200], axis=1),
        tdata=t,
        area=fit_params_guess[0:3],
        freq=fit_params_guess[3:6],
        fwhmL=fit_params_guess[6:9],
        fwhmG=fit_params_guess[9:12],
        phase=fit_params_guess[12:15],
        line_broadening=0,
        zeropad_size=np.size(t),
        method=method,
    )
    # now, iterately fit through all of the averaged frames
    data_dis_avg = signal_utils.moving_mean_2d(fids, window_size=window_size)
    data_dis_avg = data_dis_avg[:, 0:-1]
    starting_time_indices = np.arange(0, (fids.shape[1] - window_size), 1)

    fit_params0 = np.array(
        [
            fit_guess_obj.area,
            fit_guess_obj.freq,
            fit_guess_obj.fwhmL,
            fit_guess_obj.fwhmG,
            fit_guess_obj.phase,
        ]
    ).flatten()
    fit_params, snr_dyn = matlab_engine.multi_lsqcurvefit(fit_params0, t, data_dis_avg)
    # prepare the output dictionary
    out_dict = {
        constants.SpectIOFields.AREA_DYN: fit_params[:, 0:3],
        constants.SpectIOFields.FREQ_DYN: (fit_params[:, 3:6] - fit_guess_obj.freq[-1])
        / center_freq,
        constants.SpectIOFields.FWHM_DYN: fit_params[:, 6:9] / center_freq,
        constants.SpectIOFields.FWHMG_DYN: fit_params[:, 9:12] / center_freq,
        constants.SpectIOFields.PHASE_DYN: _get_positive_phase(fit_params[:, 12:15]),
        constants.SpectIOFields.SNR_DYN: snr_dyn,
        constants.SpectIOFields.T_DYN: t_tr[starting_time_indices],
    }
    return out_dict


def detrend_dynamic_data(dict_dyn: dict[str, Any]):
    """Detrend the dynamic data by subtracting the fitted exponential.

    Also high pass filter the data to remove the DC component.
    Args:
        dict_dyn (dict[str, Any]): dictionary of dynamic spectroscopy data.
            Includes the following keys:
            - AREA_DYN: area of the fit peaks.
            - FREQ_DYN: frequency of the fit peaks.
            - FWHM_DYN: FWHM of the fit peaks.
            - PHASE_DYN: phase of the fit peaks.
            - T_DYN: time points of the dynamic data.
    Returns:
        Dictionary of detrended data evaluate between 2 and 7 seconds.
    """
    b = signal_utils.get_highpass_filter(dict_dyn[constants.SpectIOFields.AREA_DYN])
    # remove the data point to avoid ringing when filtering
    t_dyn = np.delete(dict_dyn[constants.SpectIOFields.T_DYN], 0, 0)
    area_dyn = np.delete(dict_dyn[constants.SpectIOFields.AREA_DYN], 0, 0)
    freq_dyn = np.delete(dict_dyn[constants.SpectIOFields.FREQ_DYN], 0, 0)
    fwhm_dyn = np.delete(dict_dyn[constants.SpectIOFields.FWHM_DYN], 0, 0)
    phase_dyn = np.delete(dict_dyn[constants.SpectIOFields.PHASE_DYN], 0, 0)
    # get the rbc norm
    start_ind, end_ind = get_breathhold_indices(
        t=t_dyn, start_time=2, end_time=min(7, t_dyn[-1])
    )
    rbc_norm = signal_utils.get_rbc_norm_matlab(
        area_dyn[start_ind:end_ind, 0], t_dyn[start_ind:end_ind], t_dyn
    )
    # detrend by filtering and subtracting from the fitted exponential
    area_detrend = matlab_engine.filter_highpass(
        (area_dyn[:, 0] - rbc_norm) / rbc_norm
    )[start_ind:end_ind]
    freq_detrend = matlab_engine.filter_highpass(
        freq_dyn[:, 0] - np.mean(freq_dyn[start_ind:end_ind, 0])
    )[start_ind:end_ind]
    fwhm_detrend = matlab_engine.filter_highpass(
        fwhm_dyn[:, 0] - np.mean(fwhm_dyn[start_ind:end_ind, 0])
    )[start_ind:end_ind]
    phase_detrend = matlab_engine.filter_highpass(
        phase_dyn[:, 0] - np.mean(phase_dyn[start_ind:end_ind, 0])
    )[start_ind:end_ind]
    out_dict = {
        constants.SpectIOFields.AREA_DYN_DETREND: area_detrend,
        constants.SpectIOFields.FREQ_DYN_DETREND: freq_detrend,
        constants.SpectIOFields.FWHM_DYN_DETREND: fwhm_detrend,
        constants.SpectIOFields.PHASE_DYN_DETREND: phase_detrend,
        constants.SpectIOFields.T_DYN: t_dyn[start_ind:end_ind],
    }
    return out_dict


def get_oscillation_amplitude_sine(dict_detrend: dict[str, Any]):
    """Fit the detrended data to a sine wave.

    Args:
        dict_detrend (dict[str, Any]): dictionary of detrended data.
    Returns:
        Dictionary of fitted data and fit parameters.
    """
    t_dyn = dict_detrend[constants.SpectIOFields.T_DYN]
    area_dyn_fit, fit_params_area = signal_utils.fit_sine_matlab(
        dict_detrend[constants.SpectIOFields.AREA_DYN_DETREND], t_dyn
    )
    freq_dyn_fit, fit_params_freq = signal_utils.fit_sine_matlab(
        dict_detrend[constants.SpectIOFields.FREQ_DYN_DETREND], t_dyn
    )
    fwhm_dyn_fit, fit_params_fwhm = signal_utils.fit_sine_matlab(
        dict_detrend[constants.SpectIOFields.FWHM_DYN_DETREND], t_dyn
    )
    phase_dyn_fit, fit_params_phase = signal_utils.fit_sine_matlab(
        dict_detrend[constants.SpectIOFields.PHASE_DYN_DETREND], t_dyn
    )
    out_dict = {
        constants.SpectIOFields.AREA_DYN_FIT: area_dyn_fit,
        constants.SpectIOFields.FREQ_DYN_FIT: freq_dyn_fit,
        constants.SpectIOFields.FWHM_DYN_FIT: fwhm_dyn_fit,
        constants.SpectIOFields.PHASE_DYN_FIT: phase_dyn_fit,
        constants.SpectIOFields.FIT_PARAMS_SINE_AREA: fit_params_area,
        constants.SpectIOFields.FIT_PARAMS_SINE_FREQ: fit_params_freq,
        constants.SpectIOFields.FIT_PARAMS_SINE_FWHM: fit_params_fwhm,
        constants.SpectIOFields.FIT_PARAMS_SINE_PHASE: fit_params_phase,
    }
    return out_dict


def get_oscillation_amplitude_peaks(dict_detrend: dict[str, Any], data_ref: np.ndarray):
    """Fit the detrended data to a sine wave.

    Args:
        dict_detrend (dict[str, Any]): dictionary of detrended data.
        data_ref (np.ndarray): The reference data used to find the starting
            guesses in peak finding. In most cases, this will be the fit of the
            RBC oscillation area.
    Returns:
        Dictionary of fitted data and fit parameters.
    """
    t_dyn = dict_detrend[constants.SpectIOFields.T_DYN]
    tr = t_dyn[1] - t_dyn[0]
    min_peak_distance = int(68 / 60 / 6 * 1 / tr)

    def process_amplitude(data: np.ndarray, smooth: bool = False):
        indices_fitted_peaks = signal_utils.find_peaks(
            np.abs(data_ref), distance=min_peak_distance
        )
        max_values = data[indices_fitted_peaks]
        # Ensure only one point from peak and trough
        max_values = []
        indices = []
        for index_fitted_peak in indices_fitted_peaks:
            delta = np.floor_divide(np.median(np.diff(indices_fitted_peaks)), 2)
            ind_start = max(int(index_fitted_peak - delta), 1)
            ind_end = min(int(index_fitted_peak + delta), len(data))

            if smooth:
                data = signal_utils.smooth(data)
            partial_data = np.zeros_like(data)
            partial_data[ind_start:ind_end] = data[ind_start:ind_end]
            # get peaks in descending order
            index = signal_utils.find_npeaks(np.abs(partial_data), npeaks=1)
            if len(index) > 0:
                indices.append(index[0])
                max_values.append(float(data[index]))

        return np.array(max_values), np.array(indices)

    amp_area_peaks, indices_area_peaks = process_amplitude(
        dict_detrend[constants.SpectIOFields.AREA_DYN_DETREND], smooth=True
    )
    amp_freq_peaks, indices_freq_peaks = process_amplitude(
        dict_detrend[constants.SpectIOFields.FREQ_DYN_DETREND]
    )
    amp_fwhm_peaks, indices_fwhm_peaks = process_amplitude(
        dict_detrend[constants.SpectIOFields.FWHM_DYN_DETREND]
    )
    amp_phase_peaks, indices_phase_peaks = process_amplitude(
        dict_detrend[constants.SpectIOFields.PHASE_DYN_DETREND]
    )

    out_dict = {
        constants.SpectIOFields.AREA_DYN_PEAKS: amp_area_peaks,
        constants.SpectIOFields.FREQ_DYN_PEAKS: amp_freq_peaks,
        constants.SpectIOFields.FWHM_DYN_PEAKS: amp_fwhm_peaks,
        constants.SpectIOFields.PHASE_DYN_PEAKS: amp_phase_peaks,
        constants.SpectIOFields.AREA_DYN_INDICES_PEAKS: indices_area_peaks,
        constants.SpectIOFields.FREQ_DYN_INDICES_PEAKS: indices_freq_peaks,
        constants.SpectIOFields.FWHM_DYN_INDICES_PEAKS: indices_fwhm_peaks,
        constants.SpectIOFields.PHASE_DYN_INDICES_PEAKS: indices_phase_peaks,
    }
    return out_dict
