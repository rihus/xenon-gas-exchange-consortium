"""Base configuration file."""

import sys
import logging
import numpy as np
from ml_collections import config_dict

from config import config_utils

from utils import constants
# parent directory
sys.path.append("..")

class Config(config_dict.ConfigDict):
    """Base config file.

    Attributes:
        data_dir: str, path to the data directory
        manual_seg_filepath: str, path to the manual segmentation nifti file
        manual_reg_filepath: str, path to manual registration nifti file
        remove_contamination: bool, whether to remove gas contamination
        remove_noisy_projections: bool, whether to remove noisy projections
        processes: Process, the evaluation processes
        reference_data: ReferenceData, reference data
        platform: Platform, the scanner vendor platform
        reference_data_key: str, reference data key
        segmentation_key: str, the segmentation key
        hb_correction_key: str, hemoglobin correction key
        hb: float, subject hb value in g/dL
        institution: str, the scan institution
        subject_id: str, the subject id
        rbc_m_ratio: float, the RBC to M ratio
    """

    def __init__(self):
        """Initialize config parameters."""
        super().__init__()
        self.data_dir = ""
        self.subject_id = "test"
        self.data_file_name = "gx_osc_stats_kernel3"
        self.rbc_m_ratio = 0.0
        self.patient_frc = "None"
        self.bag_volume = "None"
        self.segmentation_key = constants.SegmentationKey.CNN_VENT.value
        self.manual_seg_filepath = ""
        #
        self.correction = Correction()
        self.processes = Process()
        self.reference_data_key = constants.ReferenceDataKey.DUKE_REFERENCE.value
        # self.reference_data = ReferenceData(self.reference_data_key)
        self.institution = constants.Institution.CCHMC.value
        self.system_vendor = constants.SystemVendor.PHILIPS.value
        self.recon = Recon()
        self.params = Params()
        self.vol_correction_key = constants.VolCorrectionKey.NONE.value
        self.corrected_lung_volume = "NA"
        self.dicom_proton_dir = ""
        self.multi_echo = False
        self.registration_key = constants.RegistrationKey.SKIP.value
        self.manual_reg_filepath = ""
        self.bias_key = constants.BiasfieldKey.N4ITK.value
        self.hb_correction_key = constants.HbCorrectionKey.NONE.value
        self.hb = 0.0
        self.dose = Dose()


class Correction(object):
    """Define the capillary blood volume correction process.

    Attributes:
        vc_correction: bool, whether to perform blood volume correction.
        subject_age: int, age of subject
        subject_sex: int, 1 if female or 2 if male
        subject_height: float subject height in cm
    """

    def __init__(self):
        """Initialize process parameters"""
        self.vc_correction = False
        self.subject_age = 0
        self.subject_sex = 0
        self.subject_height = 0.0


class Process(object):
    """Define the evaluation processes.

    Attributes:
        gx_mapping_recon: bool, whether to perform gas exchange mapping
            with reconstruction
        gx_mapping_readin: bool, whether to perform gas exchange mapping
            by reading in the mat file
    """

    def __init__(self):
        """Initialize the process parameters."""
        self.gx_oscillation_mapping_recon = True
        self.gx_oscillation_mapping_readin = False


class Recon(object):
    """Define reconstruction configurations.

    Attributes:
        recon_key: str, the reconstruction key
        scan_type: str, the scan type
        kernel_sharpness_lr: float, the kernel sharpness for low resolution, higher
            SNR images
        kernel_sharpness_hr: float, the kernel sharpness for high resolution, lower
            SNR images
        n_skip_start: int, the number of frames to skip at the beginning
        n_skip_end: int, the number of frames to skip at the end
        key_radius: int, the key radius for the keyhole image
    """

    def __init__(self):
        """Initialize the reconstruction parameters."""
        #Gradient delays
        self.del_x = "None"
        self.del_y = "None"
        self.del_z = "None"

        # Reconstruction and matrix sizes
        self.recon_size = 64
        self.matrix_size = 128

        self.recon_proton = False
        self.recon_key = constants.ReconKey.ROBERTSON.value
        self.scan_type = constants.ScanType.NORMALDIXON.value
        self.kernel_sharpness_lr = 0.14
        self.kernel_sharpness_hr = 0.32
        # Set initial n_skip_start value as NaN, or user input an expected value
        self.n_skip_start = config_utils.get_n_skip_start(self.scan_type)
        self.n_skip_end = 0
        self.remove_contamination = False
        self.traj_type = constants.TrajType.HALTONSPIRAL
        self.philips_software = constants.PhilipsVersion.POST_R59.value
        ## remove_noisy_projections True for Philips data pre R59, False for post R59
        if self.philips_software == constants.PhilipsVersion.PRE_R59.value:
            self.remove_noisy_projections = True
        elif self.philips_software == constants.PhilipsVersion.POST_R59.value:
            self.remove_noisy_projections = False
        elif self.philips_software == constants.PhilipsVersion.NA.value:
            logging.info("******* NOT A PHILIPS SOFTWARE. Please check that:")
            logging.info("self.remove_noisy_projections is set correctly*******")
            self.remove_noisy_projections = False


class Params(object):
    """Define important parameters.

    Attributes:
        threshold_oscillation: np.ndarray, the oscillation amplitude thresholds for
            binning
        threshold_rbc: np.ndarray, the RBC thresholds for binning
    """

    def __init__(self):
        """Initialize the reconstruction parameters."""
        self.threshold_oscillation = None
        # self.threshold_rbc = np.array([0.066, 0.250, 0.453, 0.675, 0.956]) / 2.0


class Dose(object):
    """Define dose details.

    These are not used in the pipeline, but are useful for creating statistics for
    paper writing.

    Attributes:
        de_spect: dose equivalent in ml of spectroscopy scan.
        de_dixon: dose equivalent in ml of dixon scan.
    """

    def __init__(self):
        """Initialize the scan parameters."""
        self.de_spect = 0.0
        self.de_dixon = 0.0


def get_config() -> config_dict.ConfigDict:
    """Return the config dict. This is a required function.

    Returns:
        a ml_collections.config_dict.ConfigDict
    """
    return Config()
