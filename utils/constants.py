"""Define important constants used throughout the pipeline."""

import enum

import numpy as np

FOVINFLATIONSCALE3D = 1000.0

GRYOMAGNETIC_RATIO = 11.777  # MHz/T
XENON_SHIFT = (0.3 / 5.5) * (273 / 298) * (0.548)
T2STAR_GAS = 1.8e-2  # seconds
#T2STAR_RBC_3T = 1.0502 * 1e-3  # seconds (old value)
T2STAR_RBC_3T = 1.044575 * 1e-3  # seconds
#T2STAR_MEMBRANE_3T = 1.1416 * 1e-3  # seconds (old value)
T2STAR_MEMBRANE_3T = 0.988588  * 1e-3  # seconds
T2STAR_DISSOLVED_3T = 1.5 * 1e-3  # seconds
KCO_ALPHA = 11.2  # membrane
KCO_BETA = 14.6  # RBC
VA_ALPHA = 1.43
KCO_ALPHA_MUNKHOLM = 22.3  # membrane
KCO_BETA_MUNKHOLM = 8  # RBC
VA_ALPHA_MUNKHOLM = 1.46
RBC_REF = 0.471  # normalized to gas
THETA_INV_FEMALE = 1.863  # mL_blood min mmHg / mL_CO
THETA_INV_MALE = 1.71  # mL_blood min mmHg / mL_CO
VOXEL_SIZE = 29.791e-6  # in Liters

NONE = "None"

class IOFields(object):
    """General IOFields constants."""

    BANDWIDTH = "bandwidth"
    BIASFIELD_KEY = "biasfield_key"
    BONUS_SPECTRA_LABELS = "bonus_spectra_labels"
    CONTRAST_LABELS = "contrast_labels"
    SAMPLE_TIME = "sample_time"
    FA_DIS = "fa_dis"
    FA_GAS = "fa_gas"
    FIDS = "fids"
    FIDS_DIS = "fids_dis"
    FIDS_GAS = "fids_gas"
    FIDS_BONUS_GAS = "fids_bonus_gas"
    FIDS_BONUS_DIS = "fids_bonus_dis"
    FIELD_STRENGTH = "field_strength"
    FLIP_ANGLE_FACTOR = "flip_angle_factor"
    FOV = "fov"
    XE_CENTER_FREQUENCY = "xe_center_frequency"
    XE_DISSOLVED_OFFSET_FREQUENCY = "xe_dissolved_offset_frequency"
    FREQ_CENTER = "freq_center"
    FREQ_EXCITATION = "freq_excitation"
    GIT_BRANCH = "git_branch"
    GRAD_DELAY_X = "grad_delay_x"
    GRAD_DELAY_Y = "grad_delay_y"
    GRAD_DELAY_Z = "grad_delay_z"
    HB_CORRECTION_KEY = "hb_correction_key"
    HB = "hb"
    INSTITUTION = "institution"
    SYSTEM_VENDOR = "system_vendor" #RH: scanner type
    PHILIPS_VERSION = "philips_version" #RH: philips software version
    RBC_HB_CORRECTION_FACTOR = "rbc_hb_correction_factor"
    MEMBRANE_HB_CORRECTION_FACTOR = "membrane_hb_correction_factor"
    IMAGE = "image"
    INFLATION = "inflation"
    KERNEL_SHARPNESS = "kernel_sharpness"
    N_FRAMES = "n_frames"
    N_SKIP_END = "n_skip_end"
    N_SKIP_START = "n_skip_start"
    MASK_REG_NII = "mask_reg_nii"
    N_DIS_REMOVED = "n_dis_removed"
    N_GAS_REMOVED = "n_gas_removed"
    N_POINTS = "n_points"
    NPTS = "npts"
    ORIENTATION = "orientation"
    OUTPUT_PATH = "output_path"
    PIPELINE_VERSION = "pipeline_version"
    PIXEL_SIZE = "pixel_size"
    PROCESS_DATE = "process_date"
    PROTOCOL_NAME = "protocol_name"
    PROTON_DICOM_DIR = "proton_dicom_dir"
    PROTON_REG_NII = "proton_reg_nii"
    RAMP_TIME = "ramp_time"
    RAW_PROTON_MONTAGE = "raw_proton_montage"
    REFERENCE_DATA_KEY = "reference_data_key"
    REGISTRATION_KEY = "registration_key"
    REMOVEOS = "removeos"
    REMOVE_NOISE = "remove_noise"
    SCAN_DATE = "scan_date"
    SCAN_TYPE = "scan_type"
    SEGMENTATION_KEY = "segmentation_key"
    SHAPE_FIDS = "shape_fids"
    SHAPE_IMAGE = "shape_image"
    SLICE_THICKNESS = "slice_thickness"
    SOFTWARE_VERSION = "software_version"
    SUBJECT_AGE = "subject_age"
    SUBJECT_HEIGHT = "subject_height"
    SUBJECT_ID = "subject_id"
    SUBJECT_SEX = "subject_sex"
    T2_CORRECTION_FACTOR = "t2_correction_factor"
    T2_CORRECTION_FACTOR_MEMBRANE = "t2_correction_factor_membrane"
    T2_CORRECTION_FACTOR_RBC = "t2_correction_factor_rbc"
    TE90 = "te90"
    TR = "tr"
    TR_DIS = "tr_dis"
    TRAJ = "traj"
    TRAJ_DIS = "traj_dis"
    TRAJ_DISSOVLED = "traj_dissolved"
    TRAJ_GAS = "traj_gas"
    VEN_COR_MONTAGE = "bias_cor_ven_montage"
    VEN_CV = "ven_cv"
    VEN_DEFECT = "ven_defect"
    VEN_HIGH = "ven_high"
    VEN_HIST = "ven_hist"
    VEN_LOW = "ven_low"
    VEN_MEAN = "ven_mean"
    VEN_MEDIAN = "ven_median"
    VEN_MONTAGE = "ven_montage"
    VEN_SKEW = "ven_skewness"
    VEN_SNR = "ven_snr"
    VEN_STD = "ven_std"
    VENT_DICOM_DIR = "vent_dicom_dir"
    VOL_CORRECTION_KEY = "vol_correction_key"
    VOL_CORRECTION_FACTOR_MEMBRANE  = "vol_correction_factor_membrane"
    VOL_CORRECTION_FACTOR_RBC = "vol_correction_factor_rbc"
    CORRECTED_LUNG_VOLUME = "corrected_lung_volume"
    PREP_PULSES = "prep_pulses"

class PrepPulses(enum.Enum):
    """Preparation pulse flags."""

    PREP_PULSES = "true"

class CNNPaths(object):
    """Paths to saved model files."""

class OutputPaths(object):
    """Output file names."""

    GRE_MASK_NII = "GRE_mask.nii"
    GRE_REG_PROTON_NII = "GRE_regproton.nii"
    GRE_VENT_RAW_NII = "GRE_ventraw.nii"
    GRE_VENT_COR_NII = "GRE_ventcor.nii"
    GRE_VENT_BINNING_NII = "GRE_ventbinning.nii"
    VEN_RAW_MONTAGE_PNG = "raw_ven_montage.png"
    PROTON_REG_MONTAGE_PNG = "raw_proton_montage.png"
    VEN_COR_MONTAGE_PNG = "bias_cor_ven_montage.png"
    VEN_COLOR_MONTAGE_PNG = "ven_montage.png"
    VEN_HIST_PNG = "ven_hist.png"
    HTML_TMP = "html_tmp"
    REPORT_CLINICAL = "report_clinical"


class ImageType(enum.Enum):
    """Segmentation flags."""

    VENT = "vent"
    UTE = "ute"


class SegmentationKey(enum.Enum):
    """Segmentation flags."""

    CNN_VENT = "cnn_vent"
    CNN_PROTON = "cnn_proton"
    MANUAL_VENT = "manual_vent"
    MANUAL_PROTON = "manual_proton"
    SKIP = "skip"
    THRESHOLD_VENT = "threshold_vent"


class RegistrationKey(enum.Enum):
    """Registration flags.

    Defines how and if registration is performed. Options:
    PROTON2GAS: Register ANTs to register proton image (moving) to gas image (fixed).
        Also uses the transformation and applies on the mask if segmented on proton
        image.
    MASK2GAS: Register ANTs to register mask (moving) to gas image (fixed).
        Also uses the transformation and applies on the proton image.
    MANUAL: Read in Nifti file of manually registered proton image.
    SKIP: Skip registration entirely.
    """

    MANUAL = "manual"
    MASK2GAS = "mask2gas"
    PROTON2GAS = "proton2gas"
    SKIP = "skip"


class SpectIOFields(object):
    """Spectroscopy IO Fields."""

    AREA_DYN = "area_dyn"
    AREA_DYN_DETREND = "area_dyn_detrend"
    AREA_DYN_FIT = "area_dyn_fit"
    AREA_DYN_INDICES_PEAKS = "area_dyn_indices_peaks"
    AREA_DYN_PEAKS = "area_dyn_peaks"
    DATA_DIS_AVG = "data_dis_avg"
    FIT_OBJ = "fit_obj"
    FIT_PARAMS = "fit_params"
    FIT_PARAMS_SINE_AREA = "fit_params_sine_area"
    FIT_PARAMS_SINE_FREQ = "fit_params_sine_freq"
    FIT_PARAMS_SINE_FWHM = "fit_params_sine_fwhm"
    FIT_PARAMS_SINE_PHASE = "fit_params_sine_phase"
    FREQ_DYN = "freq_dyn"
    FREQ_DYN_DETREND = "freq_dyn_detrend"
    FREQ_DYN_FIT = "freq_dyn_fit"
    FREQ_DYN_INDICES_PEAKS = "freq_dyn_indices_peaks"
    FREQ_DYN_PEAKS = "freq_dyn_peaks"
    FWHM_DYN = "fwhm_dyn"
    FWHM_DYN_DETREND = "fwhm_dyn_detrend"
    FWHM_DYN_FIT = "fwhm_dyn_fit"
    FWHM_DYN_INDICES_PEAKS = "fwhm_dyn_indices_peaks"
    FWHM_DYN_PEAKS = "fwhm_dyn_peaks"
    FWHMG_DYN = "fwhmg_dyn"
    FWHMG_DYN_DETREND = "fwhmg_dyn_detrend"
    MEMBRANE_AREA = "membrane_area"
    MEMBRANE_FWHM = "membrane_fwhm"
    MEMBRANE_FWHMG = "membrane_fwhmg"
    MEMBRANE_PHASE = "membrane_phase"
    MEMBRANE_REF = "membrane_ref"
    MEMBRANE_SHIFT_PPM = "membrane_shift"
    MEMBRANE_SNR_SNF = "membrane_snr_simulated_noise_frame"
    MEMBRANE_SNR_TAIL = "membrane_snr"
    MEMBRANE_STD = "membrane_std"
    PHASE_DYN = "phase_dyn"
    PHASE_DYN_DETREND = "phase_dyn_detrend"
    PHASE_DYN_FIT = "phase_dyn_fit"
    PHASE_DYN_INDICES_PEAKS = "phase_dyn_indices_peaks"
    PHASE_DYN_PEAKS = "phase_dyn_peaks"
    PPM = "ppm"
    PPM_SHIFT = "ppm_shift"
    RBC_AREA = "rbc_area"
    RBC_FWHM = "rbc_fwhm"
    RBC_FWHMG = "rbc_fwhmg"
    RBC_PHASE = "rbc_phase"
    RBC_REF = "rbc_ref"
    RBC_SHIFT_PPM = "rbc_shift"
    RBC_SNR_SNF = "rbc_snr_simulated_noise_frame"
    RBC_SNR_TAIL = "rbc_snr"
    RBC_STD = "rbc_std"
    SNR_DYN = "snr_dyn"
    SNRS_DYN_DETREND = "snrs_dyn_detrend"
    SPECTRUM_REF = "spectrum_ref"
    T_DYN = "t_dyn"
    T_SPECTRA = "t_spectra"


class REFERENCEFIT(object):
    """Reference fit values.

    Attributes:
        AREA (np.ndarray): Area of the reference fit.
        FREQ (np.ndarray): Frequency of the reference fit in ppm.
        FWHM (np.ndarray): Full width at half maximum of the reference fit in ppm.
        FWHMG (np.ndarray): Full width at half maximum of the reference fit in ppm.
        PHASE (np.ndarray): Phase of the reference fit.
    """

    AREA = np.array([0.60, 1.0, 0.17])
    FREQ = np.array([218.2, 197.7, 0])
    FWHM = np.array([8.72, 4.97, 1.45])
    FWHMG = np.array([0, 6.1, 0])
    PHASE = np.array([81.9, 0, 248.1])


class MatIOFields(object):
    """Mat file IO Fields."""

    SUBJECT_ID = "subject_id"
    IMAGE_RBC_OSC = "image_rbc_osc"


class BiasfieldKey(enum.Enum):
    """Biasfield correction flags.

    Defines how and if biasfield correction is performed. Options:
    N4ITK: Use N4ITK bias field correction.
    SKIP: Skip bias field ocrrection entirely.
    """

    N4ITK = "n4itk"
    SKIP = "skip"
    RF_DEPOLARIZATION = "rf_depolarization"


class ReconKey(enum.Enum):
    """Reconstruction flags.

    Options:
    ROBERTSON: scott recon
    PLUMMER: joey p. recon
    """

    ROBERTSON = "robertson"
    PLUMMER = "plummer"
    PLUMMER2 = "plummer2"


class HbCorrectionKey(enum.Enum):
    """Hb correction flags.

    Defines what level of Hb correction to apply to dissolved-phase signal. Options:
    NONE: Apply no hb correction
    RBC_AND_MEMBRANE: Apply Hb correction to both RBC and membrane signals
    RBC_ONLY: Apply Hb correction only to RBC signal
    """

    NONE = "none"
    RBC_AND_MEMBRANE = "rbc_and_membrane"
    RBC_ONLY = "rbc_only"


class VolCorrectionKey(enum.Enum):
    """Vol correction flags.
    Defines what level of volume correction to apply to dissolved-phase signal. Options:
    NONE: Apply no vol correction
    RBC_AND_MEMBRANE: Apply vol correction to both RBC and membrane signals
    """

    NONE = "False"
    RBC_AND_MEMBRANE = "True"


class ReferenceDataKey(enum.Enum):
    """Reference data flags.

    Defines which reference data to use. Options:
    DUKE_REFERENCE: Reference data for 218 or 208 ppm dissolved-phase rf excitation
    MANUAL_REFERENCE: Use when manualy adjusting default reference data
    """

    DUKE_REFERENCE = "duke_reference"
    MANUAL_REFERENCE = "manual_reference"


class ScanType(enum.Enum):
    """Scan type."""

    NORMALDIXON = "normal"
    MEDIUMDIXON = "medium"
    FASTDIXON = "fast"


class Institution(enum.Enum):
    """Institution name."""

    DUKE = "Duke"
    UVA = "UVA"
    IOWA = "University of Iowa"
    CCHMC = "CCHMC" ##RH


class SystemVendor(enum.Enum):
    """Scanner system_vendor."""

    SIEMENS = "Siemens"
    GE = "GE"    ##RH
    PHILIPS = "Philips" ##RH


class PhilipsVersion(enum.Enum):    ##RH
    """Philips scanner software version."""

    PRE_R59 = "pre_r59"
    POST_R59 = "post_r59"
    NA = "NA"


class TrajType(object):
    """Trajectory type."""

    SPIRAL = "spiral"
    HALTON = "halton"
    HALTONSPIRAL = "haltonspiral"
    SPIRALRANDOM = "spiralrandom"
    ARCHIMEDIAN = "archimedian"
    GOLDENMEAN = "goldenmean"


class Orientation(object):
    """Image orientation."""

    CORONAL = "coronal"
    AXIAL = "axial"
    TRANSVERSE = "transverse"
    CS = "cs" ##RH, for mc tag in img_utils
    NONE = "none"


class DCFSpace(object):
    """Defines the DCF space."""

    GRIDSPACE = "gridspace"
    DATASPACE = "dataspace"

class Methods(object):
    """Defines the method to calculate the RBC oscillation image."""

    ELEMENTWISE = "elementwise"
    MEAN = "mean"
    SMOOTH = "smooth"
    BSPLINE = "bspline"


class BinningMethods(object):
    """Define the method to preprocess and bin RBC oscillation image."""

    BANDPASS = "bandpass"
    FIT_SINE = "fitsine"
    NONE = "none"
    THRESHOLD_STRETCH = "threshold_stretch"
    THRESHOLD = "threshold"
    PEAKS = "peaks"
    WAVELET = "wavelet"
    MOVING_AVG = "movingavg"
    MEDIAN = "median"


class StatsIOFields(object):
    """Statistic IO Fields."""

    HEART_RATE = "heart_rate"
    INFLATION = "inflation"
    KEY_RADIUS = "key_radius"
    MEMBRANE_AREA = "membrane_area"
    MEMBRANE_FWHM_PPM = "membrane_fwhm_ppm"
    MEMBRANE_FWHMG_PPM = "membrane_fwhmg_ppm"
    MEMBRANE_NFID_SNR = "membrane_nfid_snr"
    MEMBRANE_PHASE = "membrane_phase"
    MEMBRANE_SHIFT_PPM = "membrane_shift_ppm"
    N_POINTS = "n_points"
    OSC_DEFECT_PCT = "osc_defect"
    OSC_DEFECTLOW_PCT = "osc_defectlow"
    OSC_HIGH_PCT = "osc_high"
    OSC_LOW_PCT = "osc_low"
    OSC_MEAN = "osc_mean"
    OSC_MEDIAN = "osc_median"
    OSC_STDDEV = "osc_stddev"
    PCT_OSC_NEGATIVE = "osc_negative"
    PCT_OSC_NORMAL = "osc_normal"
    PROCESS_DATE = "process_date"
    RBC_AMPLITUDE_AREA = "rbc_amplitude_area"
    RBC_AMPLITUDE_AREA_PEAKS = "rbc_amplitude_area_peaks"
    RBC_AMPLITUDE_FREQ_PPM = "rbc_amplitude_freq_ppm"
    RBC_AMPLITUDE_FREQ_PPM_PEAKS = "rbc_amplitude_freq_ppm_peaks"
    RBC_AMPLITUDE_FWHM_PPM = "rbc_amplitude_fwhm_ppm"
    RBC_AMPLITUDE_FWHM_PPM_PEAKS = "rbc_amplitude_fwhm_ppm_peaks"
    RBC_AMPLITUDE_PHASE = "rbc_amplitude_phase"
    RBC_AMPLITUDE_PHASE_PEAKS = "rbc_amplitude_phase_peaks"
    RBC_AREA = "rbc_area"
    RBC_DYN_MEAN_SNR = "rbc_dyn_mean_snr"
    RBC_FWHM_PPM = "rbc_fwhm_ppm"
    RBC_M_RATIO = "rbc_m_ratio"
    RBC_NFID_SNR = "rbc_nfid_snr"
    RBC_PHASE = "rbc_phase"
    RBC_SHIFT_PPM = "rbc_shift_ppm"
    RF_EXCITATION_PPM = "rf_excitation_ppm"
    SNR_DISSOLVED = "snr_dissolved"
    SNR_GAS = "snr_gas"
    SNR_RBC = "snr_rbc"
    SNR_RBC_HIGH = "snr_rbc_high"
    SNR_RBC_LOW = "snr_rbc_low"
    RBC_SNR = "rbc_snr"
    MEMBRANE_SNR = "membrane_snr"
    VENT_SNR = "vent_snr"
    RBC_HIGH_PCT = "rbc_high_pct"
    RBC_LOW_PCT = "rbc_low_pct"
    RBC_DEFECT_PCT = "rbc_defect_pct"
    MEMBRANE_HIGH_PCT = "membrane_high_pct"
    MEMBRANE_LOW_PCT = "membrane_low_pct"
    MEMBRANE_DEFECT_PCT = "membrane_defect_pct"
    VENT_HIGH_PCT = "vent_high_pct"
    VENT_LOW_PCT = "vent_low_pct"
    VENT_DEFECT_PCT = "vent_defect_pct"
    RBC_MEAN = "rbc_mean"
    MEMBRANE_MEAN = "membrane_mean"
    VENT_MEAN = "vent_mean"
    RBC_MEDIAN = "rbc_median"
    MEMBRANE_MEDIAN = "membrane_median"
    VENT_MEDIAN = "vent_median"
    RBC_STDDEV = "rbc_stddev"
    MEMBRANE_STDDEV = "membrane_stddev"
    VENT_STDDEV = "vent_stddev"
    DLCO_EST = "dlco_est"
    KCO_EST = "kco_est"
    ALVEOLAR_VOLUME = "alveolar_volume"

class MaskMethods(object):
    """Define methods to apply mask to an image."""

    NONE = "none"
    MIN = "min"

class VENTHISTOGRAMFields(object):
    """Ventilation histogram fields."""

    COLOR = (0.4196, 0.6824, 0.8392)
    XLIM = 1.0
    YLIM = 0.07
    NUMBINS = 50
    XTICKS = np.linspace(0, XLIM, 4)
    YTICKS = np.linspace(0, YLIM, 5)
    XTICKLABELS = ["{:.2f}".format(x) for x in XTICKS]
    YTICKLABELS = ["{:.2f}".format(x) for x in YTICKS]
    TITLE = "Ventilation"


class RBCHISTOGRAMFields(object):
    """Ventilation histogram fields."""

    COLOR = (247.0 / 255, 96.0 / 255, 111.0 / 255)
    XLIM = 0.012
    YLIM = 0.1
    NUMBINS = 50
    XTICKS = np.linspace(0, XLIM, 4)
    YTICKS = np.linspace(0, YLIM, 5)
    XTICKLABELS = ["{:.2f}".format(x * 1e2) for x in XTICKS]
    YTICKLABELS = ["{:.2f}".format(x) for x in YTICKS]
    TITLE = "RBC:Gas x 100"


class MEMBRANEHISTOGRAMFields(object):
    """Membrane histogram fields."""

    COLOR = (0.4, 0.7608, 0.6471)
    XLIM = 0.025
    YLIM = 0.18
    NUMBINS = 70
    XTICKS = np.linspace(0, XLIM, 4)
    YTICKS = np.linspace(0, YLIM, 5)
    XTICKLABELS = ["{:.2f}".format(x * 1e2) for x in XTICKS]
    YTICKLABELS = ["{:.2f}".format(x) for x in YTICKS]
    TITLE = "Membrane:Gas x 100"


class PDFOPTIONS(object):
    """PDF Options dict."""

    VEN_PDF_OPTIONS = {
        "page-width": 256,  # 320,
        "page-height": 160,  # 160,
        "margin-top": 1,
        "margin-right": 0.1,
        "margin-bottom": 0.1,
        "margin-left": 0.1,
        "dpi": 300,
        "encoding": "UTF-8",
        "enable-local-file-access": None,
    }


class NormalizationMethods(object):
    """Image normalization methods."""

    MAX = "max"
    PERCENTILE_MASKED = "percentile_masked"
    PERCENTILE = "percentile"
    MEAN = "mean"


class CMAP(object):
    """Maps of binned values to color values."""

    RBC_OSC_BIN2COLOR = {
        -1: [0.33, 0.33, 0.33],
        0: [0, 0, 0],
        1: [1, 0, 0],
        2: [1, 0.7143, 0],
        3: [0.4, 0.7, 0.4],
        4: [0, 1, 0],
        5: [184.0 / 255.0, 226.0 / 255.0, 145.0 / 255.0],
        6: [243.0 / 255.0, 205.0 / 255.0, 213.0 / 255.0],
        7: [225.0 / 255.0, 129.0 / 255.0, 162.0 / 255.0],
        8: [197.0 / 255.0, 27.0 / 255.0, 125.0 / 255.0],
    }
    RBC_BIN2COLOR = {
        0: [0, 0, 0],
        1: [1, 0, 0],
        2: [1, 0.7143, 0],
        3: [0.4, 0.7, 0.4],
        4: [0, 1, 0],
        5: [0, 0.57, 0.71],
        6: [0, 0, 1],
    }
    VENT_BIN2COLOR = {
        0: [0, 0, 0],
        1: [1, 0, 0],
        2: [1, 0.7143, 0],
        3: [0.4, 0.7, 0.4],
        4: [0, 1, 0],
        5: [0, 0.57, 0.71],
        6: [0, 0, 1],
    }
    MEMBRANE_BIN2COLOR = {
        0: [0, 0, 0],
        1: [1, 0, 0],
        2: [1, 0.7143, 0],
        3: [0.4, 0.7, 0.4],
        4: [0, 1, 0],
        5: [184.0 / 255.0, 226.0 / 255.0, 145.0 / 255.0],
        6: [243.0 / 255.0, 205.0 / 255.0, 213.0 / 255.0],
        7: [225.0 / 255.0, 129.0 / 255.0, 162.0 / 255.0],
        8: [197.0 / 255.0, 27.0 / 255.0, 125.0 / 255.0],
    }


class HbCorrection(object):
    """Coefficients for hb correction scaling factor equations.

    Reference: https://onlinelibrary.wiley.com/doi/10.1002/mrm.29712
    """

    HB_REF = 14.0  # reference hb value in g/dL
    R1 = 0.288  # coefficient of rbc hb correction equation
    M1 = 0.029  # first coefficient of membrane hb correction equation
    M2 = 0.011  # second coefficient of membrane hb correction equation

class ContrastLabels(object):
    """Numbers for labelling type of FID acquisition excitation."""

    PROTON = 0  # proton acquisition
    GAS = 1  # gas phase 129Xe acquisition
    DISSOLVED = 2  # dissolved phase 129Xe acquisition


class VolCorrection(object):
    """Coefficients for volume correction scaling factor equations
    
    Reference DOI: 10.1183/13993003.00289-2020
    """
    ALPHA_RBC = -0.15963  # slope of trend in rbc equation
    ALPHA_MEM = -0.38665   # slope of trend in membrane equation


class BonusSpectraLabels(object):
    """Numbers for labelling if FID acquisition is part of bonus spectra."""

    NOT_BONUS = 0  # not part of bonus spectra
    BONUS = 1  # part of bonus spectra

class PipelineVersion(object):
    """Pipeline version."""

    VERSION_NUMBER = 4.1 #was 4; now 4.1 With RH edits


class ReferenceDistribution(object):
    """Reference distributions for binning based on RF excitation.

    Reference: Sup's reference distribution paper when published """

    REFERENCE_218_PPM = {
    "title": "REFERENCE_218_PPM",
    "healthy_histogram_vent_dir" : "assets/histogram_profiles/218_ppm/vent_hist_profile.npy",
    "healthy_histogram_rbc_dir" : "assets/histogram_profiles/218_ppm/rbc_hist_profile.npy",
    "healthy_histogram_membrane_dir" : "assets/histogram_profiles/218_ppm/mem_hist_profile.npy",
    "threshold_vent": [0.3891, 0.5753, 0.7203, 0.8440, 0.9539],
    "threshold_rbc": [0.001393, 0.002891, 0.004772, 0.006991, 0.009518],
    "threshold_membrane": [0.004881, 0.006522, 0.008603, 0.011216, 0.014466, 0.018471, 0.023370],
    "reference_fit_vent": (0.04074, 0.7085, 0.1408),
    "reference_fit_rbc": (0.06106, 0.004942, 0.002060),
    "reference_fit_membrane": (0.0700, 0.008871, 0.002420),
    "reference_stats": {
        "vent_defect_avg": "2",
        "vent_defect_std": "",
        "vent_low_avg": "14",
        "vent_low_std": "",
        "vent_high_avg": "16",
        "vent_high_std": "",
        "membrane_defect_avg": "2",
        "membrane_defect_std": "0",
        "membrane_low_avg": "14",
        "membrane_low_std": "0",
        "membrane_high_avg": "2",
        "membrane_high_std": "0",
        "rbc_defect_avg": "2",
        "rbc_defect_std": "",
        "rbc_low_avg": "14",
        "rbc_low_std": "",
        "rbc_high_avg": "16",
        "rbc_high_std": "",
        "rbc_m_ratio_avg": "0.55",
        "rbc_m_ratio_std": "0.12",
        "inflation_avg": "3.4",
        "inflation_std": "0.33",
        "inflation_percentage":"0.0",
        "inflation_display":"0.0",
        }
    }

    REFERENCE_208_PPM = {
        "title": "REFERENCE_208_PPM",
        "healthy_histogram_vent_dir" : "assets/histogram_profiles/208_ppm/vent_hist_profile.npy",
        "healthy_histogram_rbc_dir" : "assets/histogram_profiles/208_ppm/rbc_hist_profile.npy",
        "healthy_histogram_membrane_dir" : "assets/histogram_profiles/208_ppm/mem_hist_profile.npy",
        "threshold_vent": [0.3891, 0.5753, 0.7203, 0.8440, 0.9539],
        "threshold_rbc": [0.001351, 0.002804, 0.004629, 0.006781, 0.009232],
        "threshold_membrane": [0.005320, 0.007108, 0.009377, 0.012224, 0.015766,
                                                            0.020132, 0.025471],
        "reference_fit_vent": (0.04074, 0.7085, 0.1408),
        "reference_fit_rbc": (0.06106, 0.004794, 0.001998),
        "reference_fit_membrane": (0.0700, 0.009668, 0.002638),
        "reference_stats": {
            "vent_defect_avg": "2",
            "vent_defect_std": "",
            "vent_low_avg": "14",
            "vent_low_std": "",
            "vent_high_avg": "16",
            "vent_high_std": "",
            "membrane_defect_avg": "2",
            "membrane_defect_std": "0",
            "membrane_low_avg": "14",
            "membrane_low_std": "0",
            "membrane_high_avg": "2",
            "membrane_high_std": "0",
            "rbc_defect_avg": "2",
            "rbc_defect_std": "",
            "rbc_low_avg": "14",
            "rbc_low_std": "",
            "rbc_high_avg": "16",
            "rbc_high_std": "",
            "rbc_m_ratio_avg": "0.49",
            "rbc_m_ratio_std": "0.11",
            "inflation_avg": "3.4",
            "inflation_std": "0.33",
            "inflation_percentage":"0.0",
            "inflation_display":"0.0",
            }
        }

    REFERENCE_MANUAL = {
        "title": "MANUAL",
        "healthy_histogram_vent_dir" : "assets/histogram_profiles/218_ppm/vent_hist_profile.npy",
        "healthy_histogram_rbc_dir" : "assets/histogram_profiles/218_ppm/rbc_hist_profile.npy",
        "healthy_histogram_membrane_dir" : "assets/histogram_profiles/218_ppm/mem_hist_profile.npy",
        "threshold_vent": [0.3891, 0.5753, 0.7203, 0.8440, 0.9539],
        "threshold_rbc": [0.001393, 0.002891, 0.004772, 0.006991, 0.009518],
        "threshold_membrane": [0.004881, 0.006522, 0.008603, 0.011216, 0.014466,
                                                            0.018471, 0.023370],
        "reference_fit_vent": (0.04074, 0.7085, 0.1408),
        "reference_fit_rbc": (0.06106, 0.004942, 0.002060),
        "reference_fit_membrane": (0.0700, 0.008871, 0.002420),
        "reference_stats": {
            "vent_defect_avg": "2.15",
            "vent_defect_std": "",
            "vent_low_avg": "13.59",
            "vent_low_std": "",
            "vent_high_avg": "15.74",
            "vent_high_std": "",
            "membrane_defect_avg": "2.15",
            "membrane_defect_std": "0",
            "membrane_low_avg": "13.59",
            "membrane_low_std": "0",
            "membrane_high_avg": "2.28",
            "membrane_high_std": "0",
            "rbc_defect_avg": "2.15",
            "rbc_defect_std": "",
            "rbc_low_avg": "13.59",
            "rbc_low_std": "",
            "rbc_high_avg": "15.74",
            "rbc_high_std": "",
            "rbc_m_ratio_avg": "0.59",
            "rbc_m_ratio_std": "0.12",
            "inflation_avg": "3.4",
            "inflation_std": "0.33",
            "inflation_percentage":"0.0",
            "inflation_display":"0.0",
            }
        }
