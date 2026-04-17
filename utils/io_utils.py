"""Import and export util functions."""

import os
import sys
import logging

import csv
import glob
import shutil
from typing import Any, Dict, Optional, Tuple #List,
from pathlib import Path
import ismrmrd
import mapvbvd
import nibabel as nib
import numpy as np
# import pandas as pd
import pydicom
import scipy.io as sio
from ml_collections import config_dict

from utils import constants, img_utils, mrd_utils, twix_utils

sys.path.append("..")

def import_np(path: str) -> np.ndarray:
    """Import npy file to np.ndarray.

    Args:
        path: str file path of npy file
    Returns:
        np.ndarray loaded from npy file
    """
    return np.load(path)


def import_nii(path: str) -> np.ndarray:
    """Import image as np.ndarray.

    Args:
        path: str file path of nifti file
    Returns:
        np.ndarray loaded from nifti file
    """
    return nib.load(path).get_fdata()


def import_mat(path: str) -> dict[str, Any]:
    """Import  matlab file as dictionary.

    Args:
        path: str file path of matlab file
    Returns:
        dictionary loaded from matlab file
    """
    return sio.loadmat(path)


def import_matstruct_to_dict(struct: np.ndarray) -> dict[str, Any]:
    """Import matlab struct as dictionary.

    Args:
        struct: np.ndarray matlab struct
    Returns:
        dictionary loaded from matlab file
    """
    out_dict = {}

    for field in struct.dtype.names:
        value = struct[field].flatten()[0]
        if isinstance(value[0], str):
            # strings
            out_dict[field] = str(value[0])
        elif len(value) == 1:
            # numbers
            out_dict[field] = value[0][0]
        else:
            # arrays
            out_dict[field] = np.asarray(value)
    return out_dict


def get_dyn_twix_files(path: str) -> str:
    """Get list of dynamic spectroscopy twix files.

    Args:
        path: str directory path of twix files
    Returns:
        str file path of twix file
    """
    try:
        return (
            glob.glob(os.path.join(path, "**cali**.dat"))
            + glob.glob(os.path.join(path, "**dynamic**.dat"))
            + glob.glob(os.path.join(path, "**Dynamic**.dat"))
            + glob.glob(os.path.join(path, "**dyn**.dat"))
        )[0]
    except Exception as excep:
        raise ValueError("Can't find twix file in path.") from excep


def get_dis_twix_files(path: str) -> str:
    """Get list of gas exchange twix files.

    Args:
        path: str directory path of twix files
    Returns:
        str file path of twix file
    """
    try:
        return (
            glob.glob(os.path.join(path, "**dixon***.dat"))
            + glob.glob(os.path.join(path, "**Dixon***.dat"))
        )[0]
    except Exception as excep:
        raise ValueError("Can't find twix file in path.") from excep


def get_ute_twix_files(path: str) -> str:
    """Get list of UTE twix files.

    Args:
        path: str directory path of twix files
    Returns:
        str file path of twix file
    """
    try:
        return (
            glob.glob(os.path.join(path, "**1H***.dat"))
            + glob.glob(os.path.join(path, "**BHUTE***.dat"))
            + glob.glob(os.path.join(path, "**ute***.dat"))
        )[0]
    except Exception as excep:
        raise ValueError("Can't find twix file in path.") from excep


def get_dyn_mrd_files(path: str) -> str:
    """Get list of dynamic spectroscopy MRD files.

    Args: path: str directory path of MRD files
    Returns: str file path of MRD file
    """
    try:
        return (
            glob.glob(os.path.join(path, "**Calibration***.h5"))
            + glob.glob(os.path.join(path, "**calibration***.h5"))
        )[0]
    except Exception as excep:
        raise ValueError("Can't find MRD file in path.") from excep


def get_dis_mrd_files(path: str) -> Optional[str]:
    """Get the gas-exchange MRD (.h5) file.
    Also ensures that the file name does NOT contain 'mask'.

    Args:
        path: Directory path to search (recursively).

    Returns:
        The file path of the selected MRD .h5 file, or None if no match is found.
    """
    def _no_mask(files):
        # Exclude filenames containing 'mask' (case-insensitive)
        return (f for f in files if 'mask' not in os.path.basename(f).lower())
    try:
        file_option1 = _no_mask(glob.glob(os.path.join(path,"**", "*[Gg]as*.h5"), recursive=True))
        file_option2 = _no_mask(glob.glob(os.path.join(path,"**", "*[Dd]ixon*.h5"), recursive=True))
        select_file =  (next(iter(file_option1), None) or
                        next(iter(file_option2), None))
        if select_file is None:
            logging.info("No matching file found in %s", path)
        return select_file

    except IndexError as err:
        logging.error("Error searching files in %s: %s", path, err)
        return None


def get_ute_mrd_files(path: str) -> str:
    """Get list of proton MRD files.

    Args:
        path (str): directory path of MRD files
    Returns:
        str file path of MRD file
    """
    try:
        files = glob.glob(os.path.join(path,"**", "*[Pp]roton*.h5"), recursive=True)
        if files:
            return files[0]
        files = glob.glob(os.path.join(path, "**/*ute*.h5"), recursive=True)
        if files:
            return files[0]
        files = glob.glob(os.path.join(path,"**", "*[Mm]ask*.h5"), recursive=True)
        if files:
            return files[0]
    except IndexError as errr:
        logging.error("can't find ute file: %s", errr)


def get_mat_file(path: str) -> str:
    """Get list of mat file of reconstructed images.

    Args:
        path: str directory path of mat file.
    Returns:
        str file path of mat file
    """
    try:
        return (glob.glob(os.path.join(path, "**.mat")))[0]
    except Exception as exc:
        raise ValueError(f"Can't find mat file in path: {path}") from exc


def read_dyn_twix(path: str) -> dict[str, Any]:
    """Read dynamic spectroscopy twix file.

    Args:
        path: str file path of twix file
    Returns: dictionary containing data and metadata extracted from the twix file.
    This includes:
        1. scan date in MM-DD-YYY format.
        2. dwell time in seconds.
        3. TR in seconds.
        4. Center frequency in MHz.
        5. excitation frequency in ppm.
        6. dissolved phase FIDs in format (n_points, n_projections).
    """
    try:
        twix_obj = mapvbvd.mapVBVD(path)
    except Exception as exc:
        raise ValueError("Invalid twix file.") from exc
    twix_obj.image.squeeze = True
    twix_obj.image.flagIgnoreSeg = True
    twix_obj.image.flagRemoveOS = False

    # Get scan information
    sample_time = twix_utils.get_dwell_time(twix_obj=twix_obj)
    fids_dis = twix_utils.get_dyn_fids(twix_obj)
    freq_center = twix_utils.get_center_freq(twix_obj=twix_obj)
    freq_excitation = twix_utils.get_excitation_freq(twix_obj=twix_obj)
    scan_date = twix_utils.get_scan_date(twix_obj=twix_obj)
    tr = twix_utils.get_TR(twix_obj=twix_obj)

    return {
        constants.IOFields.SAMPLE_TIME: sample_time,
        constants.IOFields.FIDS_DIS: fids_dis,
        constants.IOFields.FREQ_CENTER: freq_center,
        constants.IOFields.FREQ_EXCITATION: freq_excitation,
        constants.IOFields.SCAN_DATE: scan_date,
        constants.IOFields.TR: tr,
    }


def read_dis_twix(path: str) -> dict[str, Any]:
    """Read 1-point dixon disssolved phase imaging twix file.

    Args:
        path: str file path of twix file
    Returns: dictionary containing data and metadata extracted from the twix file.
    This includes:
        - dwell time in seconds.
        - flip angle applied to dissolved phase in degrees.
        - flip angle applied to gas phase in degrees.
        - dissolved phase FIDs in format (n_projections, n_points).
        - gas phase FIDs in format (n_projections, n_points).
        - field of view in cm.
        - center frequency in MHz.
        - excitation frequency in ppm.
        - gradient delay (x, y, z) in microseconds.
        - number of frames (projections) used to calculate trajectory.
        - number of projections to skip at the beginning of the scan.
        - number of projections to skip at the end of the scan.
        - orientation of the scan.
        - protocol name
        - ramp time in microseconds.
        - scan date in YYYY-MM-DD format.
        - software version
        - TE90 in seconds.
        - TR in seconds.
    """
    try:
        twix_obj = mapvbvd.mapVBVD(path)
    except Exception as excep:
        raise ValueError("Invalid twix file.") from excep
    twix_obj.image.squeeze = True
    twix_obj.image.flagIgnoreSeg = True
    twix_obj.image.flagRemoveOS = False

    data_dict = twix_utils.get_gx_data(twix_obj=twix_obj)
    filename = os.path.basename(path)

    return {
        constants.IOFields.SAMPLE_TIME: twix_utils.get_dwell_time(twix_obj),
        constants.IOFields.FA_DIS: twix_utils.get_flipangle_dissolved(twix_obj),
        constants.IOFields.FA_GAS: twix_utils.get_flipangle_gas(twix_obj),
        constants.IOFields.FIDS: data_dict[constants.IOFields.FIDS],
        constants.IOFields.FIDS_DIS: data_dict[constants.IOFields.FIDS_DIS],
        constants.IOFields.FIDS_GAS: data_dict[constants.IOFields.FIDS_GAS],
        constants.IOFields.FIELD_STRENGTH: twix_utils.get_field_strength(twix_obj),
        constants.IOFields.FOV: twix_utils.get_FOV(twix_obj),
        constants.IOFields.FREQ_CENTER: twix_utils.get_center_freq(twix_obj),
        constants.IOFields.FREQ_EXCITATION: twix_utils.get_excitation_freq(twix_obj),
        constants.IOFields.GRAD_DELAY_X: data_dict[constants.IOFields.GRAD_DELAY_X],
        constants.IOFields.GRAD_DELAY_Y: data_dict[constants.IOFields.GRAD_DELAY_Y],
        constants.IOFields.GRAD_DELAY_Z: data_dict[constants.IOFields.GRAD_DELAY_Z],
        constants.IOFields.N_FRAMES: data_dict[constants.IOFields.N_FRAMES],
        constants.IOFields.N_SKIP_END: data_dict[constants.IOFields.N_SKIP_END],
        constants.IOFields.N_SKIP_START: data_dict[constants.IOFields.N_SKIP_START],
        constants.IOFields.ORIENTATION: twix_utils.get_orientation(twix_obj),
        constants.IOFields.PROTOCOL_NAME: twix_utils.get_protocol_name(twix_obj),
        constants.IOFields.RAMP_TIME: twix_utils.get_ramp_time(twix_obj),
        constants.IOFields.REMOVEOS: twix_utils.get_flag_removeOS(twix_obj),
        constants.IOFields.SCAN_DATE: twix_utils.get_scan_date(twix_obj),
        constants.IOFields.SOFTWARE_VERSION: twix_utils.get_software_version(twix_obj),
        constants.IOFields.TE90: twix_utils.get_TE90(twix_obj),
        constants.IOFields.TR: twix_utils.get_TR_dissolved(twix_obj),
        constants.IOFields.BANDWIDTH: twix_utils.get_bandwidth(
            twix_obj, data_dict, filename
        ),
    }


def read_bonus_twix(path: str) -> dict[str, Any]:
    """Read the bonus spectrum at the end of the dissolved scan.

    Args:
        path: str file path of twix file
    Returns:
        dictionary containing data and metadata extracted from the twix file.
    """
    try:
        twix_obj = mapvbvd.mapVBVD(path)
    except Exception as exc:
        raise ValueError("Invalid twix file.") from exc
    twix_obj.image.squeeze = True
    twix_obj.image.flagIgnoreSeg = True
    twix_obj.image.flagRemoveOS = False
    data_dict = twix_utils.get_bonus_data(twix_obj)
    return data_dict


def read_ute_twix(path: str) -> dict[str, Any]:
    """Read proton ute imaging twix file.

    Args:
        path: str file path of twix file
    Returns: dictionary containing data and metadata extracted from the twix file.
    This includes:
        TODO
    """
    try:
        twix_obj = mapvbvd.mapVBVD(path)
    except Exception as excep:
        raise ValueError("Invalid twix file.") from excep
    try:
        twix_obj.image.squeeze = True
    except:
        # this is old data, need to get the 2nd element
        twix_obj = twix_obj[1]
    try:
        twix_obj.image.squeeze = True
        twix_obj.image.flagIgnoreSeg = True
        twix_obj.image.flagRemoveOS = False
    except Exception as excep:
        raise ValueError("Cannot get data from twix object.") from excep
    data_dict = twix_utils.get_ute_data(twix_obj=twix_obj)

    return {
        constants.IOFields.SAMPLE_TIME: twix_utils.get_dwell_time(twix_obj),
        constants.IOFields.FIDS: data_dict[constants.IOFields.FIDS],
        constants.IOFields.RAMP_TIME: twix_utils.get_ramp_time(twix_obj),
        constants.IOFields.GRAD_DELAY_X: data_dict[constants.IOFields.GRAD_DELAY_X],
        constants.IOFields.GRAD_DELAY_Y: data_dict[constants.IOFields.GRAD_DELAY_Y],
        constants.IOFields.GRAD_DELAY_Z: data_dict[constants.IOFields.GRAD_DELAY_Z],
        constants.IOFields.N_FRAMES: data_dict[constants.IOFields.N_FRAMES],
        constants.IOFields.ORIENTATION: twix_utils.get_orientation(twix_obj),
    }


def read_dyn_mrd(path: str) -> dict[str, Any]:
    """Read dynamic spectroscopy MRD file.

    Args:
        path: str file path of MRD file
    Returns: dictionary containing data and metadata extracted from the MRD file.
    This includes:
        1. scan date in MM-DD-YYY format.
        2. dwell time in seconds.
        3. TR in seconds.
        4. Center frequency in MHz.
        5. excitation frequency in ppm.
        6. dissolved phase FIDs in format (n_points, n_projections).
    """
    try:
        dataset = ismrmrd.Dataset(path, "dataset", create_if_needed=False)
        header = ismrmrd.xsd.CreateFromDocument(dataset.read_xml_header())
    except Exception as excep:
        raise ValueError("Invalid mrd file.") from excep
    # Get scan information
    sample_time = mrd_utils.get_sample_time(dataset=dataset)
    fids_dis = mrd_utils.get_dyn_fids(dataset=dataset)
    xe_center_frequency = mrd_utils.get_center_freq(header=header)
    xe_dissolved_offset_frequency = mrd_utils.get_excitation_freq(header=header)
    scan_date = mrd_utils.get_scan_date(header=header)
    tr = mrd_utils.get_TR(header=header)

    return {
        constants.IOFields.SAMPLE_TIME: sample_time,
        constants.IOFields.FIDS_DIS: fids_dis,
        constants.IOFields.XE_CENTER_FREQUENCY: xe_center_frequency,
        constants.IOFields.XE_DISSOLVED_OFFSET_FREQUENCY: xe_dissolved_offset_frequency,
        constants.IOFields.SCAN_DATE: scan_date,
        constants.IOFields.TR: tr,
    }


def read_dis_mrd(path: str, multi_echo: bool) -> dict[str, Any]: #
    """Read 1-point dixon disssolved phase imaging mrd file.

    Args:
        path: str file path of mrd file
        multi_echo: option to perform multi echo
    Returns: dictionary containing data and metadata extracted from the mrd file.
    This includes:
        - dwell time in seconds.
        - flip angle applied to dissolved phase in degrees.
        - flip angle applied to gas phase in degrees.
        - dissolved phase FIDs in format (n_projections, n_points).
        - gas phase FIDs in format (n_projections, n_points).
        - field of view in cm.
        - center frequency in MHz.
        - excitation frequency in ppm.
        - gradient delay (x, y, z) in microseconds.
        - number of frames (projections) used to calculate trajectory.
        - number of projections to skip at the beginning of the scan.
        - number of projections to skip at the end of the scan.
        - orientation of the scan.
        - protocol name
        - ramp time in microseconds.
        - scan date in YYYY-MM-DD format.
        - software version
        - TE90 in seconds.
        - TR in seconds.
    """
    try:
        dataset = ismrmrd.Dataset(path, "dataset", create_if_needed=False)
        header = ismrmrd.xsd.CreateFromDocument(dataset.read_xml_header())
    except Exception as excep:
        raise ValueError("Invalid mrd file.") from excep

    data_dict = mrd_utils.get_gx_data(dataset, multi_echo) #

    return {
        constants.IOFields.SUBJECT_AGE: np.nan,
        constants.IOFields.SUBJECT_SEX: np.nan,
        constants.IOFields.SUBJECT_HEIGHT: np.nan,
        constants.IOFields.BANDWIDTH: np.nan,
        constants.IOFields.SAMPLE_TIME: mrd_utils.get_sample_time(dataset),
        constants.IOFields.FA_DIS: mrd_utils.get_flipangle_dissolved(header),
        constants.IOFields.FA_GAS: mrd_utils.get_flipangle_gas(header),
        constants.IOFields.FIDS: data_dict[constants.IOFields.FIDS],
        constants.IOFields.FIDS_DIS: data_dict[constants.IOFields.FIDS_DIS],
        constants.IOFields.FIDS_GAS: data_dict[constants.IOFields.FIDS_GAS],
        constants.IOFields.FIELD_STRENGTH: mrd_utils.get_field_strength(header),
        constants.IOFields.FOV: mrd_utils.get_FOV(header),
        constants.IOFields.XE_CENTER_FREQUENCY: mrd_utils.get_center_freq(header),
        constants.IOFields.XE_DISSOLVED_OFFSET_FREQUENCY: mrd_utils.get_excitation_freq(
            header),
        constants.IOFields.GRAD_DELAY_X: np.nan, #RH: updated from new code
        constants.IOFields.GRAD_DELAY_Y: np.nan,
        constants.IOFields.GRAD_DELAY_Z: np.nan,
        constants.IOFields.INSTITUTION: mrd_utils.get_institution_name(header),
        constants.IOFields.SYSTEM_VENDOR: mrd_utils.get_system_vendor(header),
        constants.IOFields.ORIENTATION: mrd_utils.get_orientation(header),
        constants.IOFields.PROTOCOL_NAME: mrd_utils.get_protocol_name(header),
        constants.IOFields.RAMP_TIME: mrd_utils.get_ramp_time(header),
        constants.IOFields.REMOVEOS: False,
        constants.IOFields.SCAN_DATE: mrd_utils.get_scan_date(header),
        constants.IOFields.SOFTWARE_VERSION: "NA",
        constants.IOFields.TE90: mrd_utils.get_TE90(header)/1000.0, #
        constants.IOFields.TR: mrd_utils.get_TR(header=header),
        constants.IOFields.TR_DIS: mrd_utils.get_TR_dissolved(header),
        constants.IOFields.TRAJ: data_dict[constants.IOFields.TRAJ],
        constants.IOFields.PREP_PULSES: mrd_utils.get_prep_pulses(header),
        # "matrix_size": data_dict["matrix_size"]
    }

def read_ute_mrd(path: str) -> Dict[str, Any]:
    """Read proton MRD file.

    Args:
        path (str): file path of mrd file
    Returns: dictionary containing data and metadata extracted from the mrd file.
    This includes:
        TODO
    """
    try:
        dataset = ismrmrd.Dataset(path, "dataset", create_if_needed=False)
        header = ismrmrd.xsd.CreateFromDocument(dataset.read_xml_header())
    except Exception as excep:
        raise ValueError("Invalid mrd file.") from excep

    data_dict = mrd_utils.get_ute_data(dataset)
    return {
        constants.IOFields.SAMPLE_TIME: mrd_utils.get_sample_time(dataset),
        constants.IOFields.FIDS: data_dict[constants.IOFields.FIDS],
        constants.IOFields.ORIENTATION: mrd_utils.get_orientation(header),
        constants.IOFields.RAMP_TIME: mrd_utils.get_ramp_time(header),
        constants.IOFields.GRAD_DELAY_X: np.nan,
        constants.IOFields.GRAD_DELAY_Y: np.nan,
        constants.IOFields.GRAD_DELAY_Z: np.nan,
        constants.IOFields.TRAJ: data_dict[constants.IOFields.TRAJ],
    }


def read_dicom(path: str, shape_gas: Tuple[int, int, int]) -> np.ndarray:
    """Read in DICOM image as numpy array.

    Args:
        path (str): path to DICOM proton image
        shape_gas (tuple[int, int, int]): size of Dixon gas image

    Returns:
        np.ndarray: numpy array from DICOM image.
    """
    files = os.listdir(path)
    file1 = os.path.join(path, files[0])
    RefDs = pydicom.dcmread(file1)
    ConstPixelDims = (int(RefDs.Columns), int(RefDs.Rows), len(files))
    dicom = np.zeros(ConstPixelDims)

    for i in range(0, len(files)):
        if files[i] == ".DS_Store":
            continue
        file_with_path = os.path.join(path, files[i])
        RefDs = pydicom.dcmread(file_with_path)
        slice_num = int(RefDs.InstanceNumber)
        ds = pydicom.read_file(file_with_path)
        dicom[:, :, slice_num - 1] = ds.pixel_array

    image = dicom.astype("float64")

    if image.shape[0] != shape_gas[0]:
        image = img_utils.interp(image, shape_gas[0] / image.shape[0])

    return image


def load_manual_segmentation(mask_path):
    """
    Load a manual segmentation NIfTI file (.nii.gz or .nii).
    
    If the given path doesn't exist:
      - If it ends with .nii.gz -> try the same name with .nii
      - If it ends with .nii    -> try the same name with .nii.gz
      - Otherwise               -> try appending .nii.gz then .nii

    Parameters
    ----------
    mask_path : str or pathlib.Path
        Path to the segmentation file (expected .nii.gz or .nii).

    Returns
    -------
    np.ndarray (bool)
        The segmentation as a boolean array, squeezed.

    Raises
    ------
    FileNotFoundError
        If neither the original nor the alternate file exists.
    """
    p = Path(mask_path)
    s = p.as_posix()

    # Build candidate paths to try in order
    if s.endswith('.nii.gz'):
        candidates = [p, Path(s[:-3])]  # remove the trailing '.gz' -> .nii
    elif s.endswith('.nii'):
        candidates = [p, Path(s + '.gz')]  # -> .nii.gz
    else:
        # If no (recognized) suffix given, try both extensions
        candidates = [p, Path(s + '.nii.gz'), Path(s + '.nii')]

    # Deduplicate while preserving order
    seen, unique_candidates = set(), []
    for c_ in candidates:
        if c_ not in seen:
            unique_candidates.append(c_)
            seen.add(c_)

    # Try to load the first existing file
    for c_ in unique_candidates:
        if c_.is_file():
            logging.info("Mask file loading: %s", c_)
            data_ = nib.load(str(c_)).get_fdata()
            return np.squeeze(np.asarray(data_)).astype(bool)

    tried = ", ".join(str(c_) for c_ in unique_candidates)
    raise FileNotFoundError(f"Manual segmentation not found. Tried: {tried}")


def export_nii(image: np.ndarray, path: str, fov: Optional[float] = None):
    """Export image as nifti file.

    Args:
        image: np.ndarray 3D image to be exporetd
        path: str file path of nifti file
        fov: float field of view
    """
    nii_imge = nib.Nifti1Image(image, np.eye(4))
    if fov:
        nii_imge.header["pixdim"][1:4] = [
            fov / np.shape(image)[0] / 10,
            fov / np.shape(image)[0] / 10,
            fov / np.shape(image)[0] / 10,
        ]
    nib.save(nii_imge, path)


def export_nii_4d(image: np.ndarray, path: str, fov: Optional[float] = None):
    """Export 4d image image as nifti file.

    Args:
        image (np.ndarray): 4D image to be exported of shape (x, y, z, 3)
        path (str): file path of nifti file
        fov (Optional[float], optional): field of view. Defaults to None.
    """
    color = (np.copy(image) * 255).astype("uint8")  # need uint8 to save to RGB
    # some fancy and tricky re-arrange
    color = np.transpose(color, [2, 3, 0, 1])
    cline = np.reshape(color, (1, np.size(color)))
    color = np.reshape(cline, np.shape(image), order="A")
    color = np.transpose(color, [2, 1, 0, 3])
    # stake the RGB channels
    shape_3d = image.shape[0:3]
    rgb_dtype = np.dtype([("R", "u1"), ("G", "u1"), ("B", "u1")])
    nii_data = (
        color.copy().view(dtype=rgb_dtype).reshape(shape_3d)
    )  # copy used to force fresh internal structure

    # write voxel dimensions to nifti header if available
    nii_imge = nib.Nifti1Image(nii_data, np.eye(4))
    if fov:
        nii_imge.header["pixdim"][1:4] = [
            fov / np.shape(image)[0] / 10,
            fov / np.shape(image)[0] / 10,
            fov / np.shape(image)[0] / 10,
        ]
    nib.save(nii_imge, path)


def export_subject_mat(subject: object, path: str):
    """Export select subject instance variables to mat file.

    Args:
        subject: subject instance
        path: str file path of mat file
    """
    sio.savemat(path, vars(subject))


def export_np(arr: np.ndarray, path: str):
    """Export numpy array to npy file.

    Args:
        arr: np.ndarray array to be exported
        path: str file path of npy file
    """
    np.save(path, arr)


def export_subject_csv(dict_stats: Dict[str, Any], path: str, overwrite=False):
    """Export statistics to running csv file.

    Uses the csv.DictWriter class to write a csv file. First, checks if the csv
    file exists and the header has been written. If not, writes the header. Then,
    writes to a new file or new row of data in existing file.

    Args:
        dict_stats (dict): dictionary containing statistics to be exported
        path (str): file path of csv file
        overwrite (bool): if True, overwrite existing csv file
    """
    header = dict_stats.keys()
    if overwrite or (not os.path.exists(path)):
        with open(path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=header)
            writer.writeheader()
            writer.writerow(dict_stats)
    else:
        with open(path, "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=header)
            writer.writerow(dict_stats)


def export_config_to_json(config: config_dict, path: str) -> None:
    """
    Save a dictionary to a JSON file.

    Args:
    - data (ml_collections.ConfigDict): The config dictionary to save.
    - path (str): The name of the file to save the dictionary to.

    Returns:
    - None
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write(config.to_json_best_effort(indent=4))


def move_files(source_paths: list, destination_path: str) -> None:
    """Move files to a new directory.

    If target directory does not exist, it is created.
    Args:
        source_paths (list): list of paths of files to move
        destination_path (str): path to move files to
    """
    # if target directory doesn't exist, create it
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)
    # move files to target directory
    for path in source_paths:
        fname = os.path.basename(path)
        if os.path.isfile(path):
            shutil.move(path, os.path.join(destination_path, fname))
