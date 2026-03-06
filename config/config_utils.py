"""Util functions for config files."""

import numpy as np

from utils import constants

def get_n_skip_start(scan_type: str) -> int:
    """Get the number of frames to skip at the beginning of the dissolved phase scan.

    Args:
        scan_type: str, the scan type
    Returns:
        the number of frames to skip at the beginning
    """
    if scan_type == constants.ScanType.NORMALDIXON.value:
        return 60
    elif scan_type == constants.ScanType.MEDIUMDIXON.value:
        return 60
    elif scan_type == constants.ScanType.FASTDIXON.value:
        return 200
    else:
        raise ValueError(f"Scan type: {scan_type} is not recognized.")

def get_thresholds(recon_key: str) -> np.ndarray:
    """Thresholds for RBC Oscillations Amplitudes"""
    if recon_key == constants.ReconKey.PLUMMER.value:
        return np.array([-0.38, 1.73, 4.41, 7.85, 12.31, 18.16, 25.93])
    if recon_key == constants.ReconKey.ROBERTSON.value:
        return np.array([-2.02, 0.53, 3.66, 7.63, 12.99, 21.07, 35.56])
    raise ValueError(f"Invalid scan type: {recon_key}")
