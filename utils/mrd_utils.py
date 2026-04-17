"""MRD util functions."""
import logging
import sys
from typing import Any, Dict

import ismrmrd
import numpy as np

from utils import constants, signal_utils
sys.path.append("..")

def get_subject_id(
    header: ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader,
) -> str:
    """Get subject ID from the MRD header.

    Args
        header (ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader): MRD header
    Returns:
        subject ID (str)
    """
    return header.subjectInformation.patientID


def get_system_vendor(
    header: ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader,
) -> str:
    """Get system vendor from the MRD header.

    Args
        header (ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader): MRD header
    Returns:
        system vendor (str)
    """
    return header.acquisitionSystemInformation.systemVendor


def get_institution_name(
    header: ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader,
) -> str:
    """Get the institution name from the MRD header.

    Args:
        header (ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader): MRD header
    Returns:
        str: institution name
    """
    return header.acquisitionSystemInformation.institutionName


def get_field_strength(
    header: ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader,
) -> float:
    """Get the magnetic field strength from the MRD header in Tesla.

    Args:
        header (ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader): MRD header
    Returns:
        magnetic field strength in Tesla (float)
    """
    return header.acquisitionSystemInformation.systemFieldStrength_T


def get_sample_time(dataset: ismrmrd.hdf5.Dataset) -> float:
    """Get the sample time from the MRD data set object.

    Sample time is stored for every FID acquisition. Assumes sample time is the same
    for each acquisition and reads sample time from header of first acquisition.

    Args:
        dataset (ismrmrd.hdf5.Dataset): MRD data object
    Returns:
        float: dwell time in seconds
    """
    acq_header = dataset.read_acquisition(0).getHead()
    return acq_header.sample_time_us * 1e-6


def get_sample_time_gas_exchange(dataset: ismrmrd.hdf5.Dataset) -> float:
    """
    Get the sample (dwell) time from the MRD dataset.

    Reads from the first non-bonus spectrum acquisition.

    Args:
        dataset (ismrmrd.hdf5.Dataset): MRD data object
    Returns:
        float: dwell time in seconds
    """
    n_acq = dataset.number_of_acquisitions()
    for i in range(n_acq):
        acq = dataset.read_acquisition(i)
        head = acq.getHead()

        if head.measurement_uid != 1:
            return head.sample_time_us * 1e-6

    raise RuntimeError("No valid acquisitions found to determine sample time.")


def get_sample_time_bonus_spectra(dataset: ismrmrd.hdf5.Dataset) -> float:
    """
    Get the sample (dwell) time for bonus spectra from the MRD dataset.

    Uses the first acquisition identified as a bonus spectrum.

    Args:
        dataset (ismrmrd.hdf5.Dataset): MRD data object

    Returns:
        float: dwell time in seconds
    """
    n_acq = dataset.number_of_acquisitions()
    for i in range(n_acq):
        acq = dataset.read_acquisition(i)
        head = acq.getHead()

        if head.measurement_uid:
            return head.sample_time_us * 1e-6

    raise RuntimeError("No bonus spectra acquisitions found to determine sample time.")


def get_dyn_fids(dataset: ismrmrd.hdf5.Dataset, n_skip_end: int = 20) -> np.ndarray:
    """Get the dissolved phase FIDS used for dyn. spectroscopy from mrd object.

    Args:
        header (ismrmrd.hdf5.Dataset): MRD dataset
        n_skip_end: number of fids to skip from the end. Usually they are calibration
            frames.
    Returns:
        dissolved phase FIDs in shape (number of points in ray, number of projections).
    """
    raw_fids = []
    n_projections = dataset.number_of_acquisitions() - n_skip_end
    for i in range(0, int(n_projections)):  # type: ignore
        raw_fids.append(dataset.read_acquisition(i).data[0].flatten())
    return np.transpose(np.asarray(raw_fids))


def get_excitation_freq(
    header: ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader,
) -> float:
    """Get the excitation frequency from the MRD header.

    Args:
        header (ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader): MRD header

    Returns:
        excitation frequency in ppm (float)
    """
    try:
        gasExciFreq = header.encoding[0].trajectoryDescription.userParameterDouble[2].value  # in Hz
        disExciFreq = header.encoding[0].trajectoryDescription.userParameterDouble[3].value  # in Hz
        excitation = disExciFreq - gasExciFreq

        gyro_ratio = 11.777  # gyromagnetic ratio of 129Xe in MHz/Tesla
        freq_excitation_ppm= round(
            excitation / (gyro_ratio * header.acquisitionSystemInformation.systemFieldStrength_T), 1
        )
    except:
        try:
            freq_excitation_ppm= round(
                header.encoding[0].userParameters.userParameterDouble[0].value
            )
        except:
            freq_excitation_ppm = 208
    return freq_excitation_ppm

def get_center_freq(header: ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader) -> float:
    """Get the center frequency from the MRD header.

    See: https://mriquestions.com/center-frequency.html for definition of center freq.
    Args:
        header (ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader): MRD header

    Returns:
        center frequency in MHz (float)
    """

    try:
        center_Xe = header.encoding[0].trajectoryDescription.userParameterDouble[2].value;
    except:

        center_Xe =353371150.0

    return center_Xe  * 1e-6


def get_TR(header: ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader) -> float:
    """Get the TR from the MRD header.

    Args:
        header (ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader): MRD header

    Returns:
        float: TR in seconds
    """
    return 1e-3 * header.sequenceParameters.TR[0]


def get_scan_date(header: ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader) -> str:
    """Get the scan date from the MRD header in MM-DD-YYYY format.

    Args:
        header (ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader): MRD header

    Returns:
        str: scan date in MM-DD-YYYY format.
    """
    xml_date = header.measurementInformation.seriesDate
    MM = "0" + str(xml_date[0]) if len(str(xml_date[0])) == 1 else str(xml_date[0])
    DD = "0" + str(xml_date[1]) if len(str(xml_date[1])) == 1 else str(xml_date[1])
    YYYY = str(xml_date[2])
    return MM + "-" + DD + "-" + YYYY


def get_flipangle_dissolved(
    header: ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader,
) -> float:
    """Get the dissolved phase flip angle in degrees.

    Args:
        header (ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader): MRD header
    Returns:
        flip angle in degrees
    """
    return header.sequenceParameters.flipAngle_deg[1]


def get_flipangle_gas(header: ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader) -> float:
    """Get the gasd phase flip angle in degrees.

    Args:
        header (ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader): MRD header
    Returns:
        flip angle in degrees
    """
    return header.sequenceParameters.flipAngle_deg[0]


def get_prep_pulses(
    header: ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader,
) -> str:
    """Get the prep pulse.

    Args:
        header (ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader): MRD header
    Returns:
        prep_pulse (string)
    """
    var_names = [
        up.name for up in header.userParameters.userParameterString
    ]
    # Check if PREP_PULSES exists
    if constants.IOFields.PREP_PULSES not in var_names:
        return "prep_pulses does not exist in the MRD file."

    prep_pulses = str(
        header.userParameters.userParameterString[
            var_names.index(constants.IOFields.PREP_PULSES)
        ]
        .value
    )
    return prep_pulses

def get_FOV(header: ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader) -> float:
    """Get the FOV in cm.

    For now, assumes same FOV in all three dimensions.

    Args:
        header (ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader): MRD header
    Returns:
        FOV in cm (float)
    """
    return header.encoding[0].reconSpace.fieldOfView_mm.x * 1e-1


def get_orientation(header: ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader) -> str:
    """Get the orientation of the image.

    Args:
        header (ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader): MRD header
    Returns:
        orientation (str): orientation of reconstructed image (coronal, transverse, axial).
            Returns coronal if not found.
    """
    orientation = ""
    system_vendor = get_system_vendor(header)

    try:
        var_names = [
            header.userParameters.userParameterString[i].name
            for i in range(len(header.userParameters.userParameterString))
        ]
        orientation = header.userParameters.userParameterString[
            var_names.index(constants.IOFields.ORIENTATION)
        ].value
    except:
        logging.info("Unable to find orientation from twix object, returning coronal.")

    if system_vendor == constants.SystemVendor.PHILIPS.value:
        if orientation.lower() == constants.Orientation.CORONAL or not orientation:
            return constants.Orientation.CORONAL
    elif system_vendor == constants.SystemVendor.GE.value:
        if orientation.lower() == constants.Orientation.CORONAL or not orientation:
            return constants.Orientation.CORONAL
    else:
        if orientation.lower() == constants.Orientation.CORONAL or not orientation:
            return constants.Orientation.CORONAL


def get_protocol_name(header: ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader) -> str:
    """Get the protocol name.

    Args:
        header (ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader): MRD header
    Returns:
        protocol name. Returns "unknown" if not found.
    """
    try:
        return str(header.measurementInformation.protocolName)
    except:
        return "unknown"


def get_ramp_time(header: ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader) -> float:
    """Get the ramp time in micro-seconds.

    See: https://mriquestions.com/gradient-specifications.html

    Args:
        header (ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader): MRD header
    Returns:
        ramp time in us
    """
    var_names = [
        header.encoding[0].trajectoryDescription.userParameterLong[i].name
        for i in range(len(header.encoding[0].trajectoryDescription.userParameterLong))
    ]
    ramp_time = float(
        header.encoding[0]
        .trajectoryDescription.userParameterLong[
            var_names.index(constants.IOFields.RAMP_TIME)
        ]
        .value
    )
    return ramp_time


def get_TE90(header: ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader) -> float:
    """Get the TE90 in seconds.

    Args:
        header (ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader): MRD header
    Returns:
        TE90 in seconds
    """
    return header.sequenceParameters.TE[0] # * 1e-3 commented RH

def get_TR_dissolved(header: ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader) -> float:
    """Get the TR in seconds for dissolved phase.

    The dissolved phase TR is defined to be the time between two consecutive dissolved
    phase-FIDS. This is different from the TR in the mrd header as the mrd header
    provides the dissolved and gas phase interleaf durations.

    Args:
        header (ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader): MRD header
    Returns:
        TR in seconds (float)
    """
    tr_gas_to_dissolved = header.sequenceParameters.TR[0]
    # tr_dissolved_to_gas = header.sequenceParameters.TR[1] ##TODO cound't read this: RH
    return (2 * tr_gas_to_dissolved) * 1e-3 # + tr_dissolved_to_gas


def get_gx_data(dataset: ismrmrd.hdf5.Dataset, multi_echo: bool) -> Dict[str, Any]:
    """Get the FID acquisition data from dixon MRD file.

    Args:
        dataset: ismrmrd dataset object
    Returns:
        a dictionary containing
            - all raw fids of shape (number of projections for gas and dissolved phase combined,
                number of points in ray)
            - gas phase fids in shape (number of projections, number of points in ray)
            - dissolved phase fids in shape (number fo projections, number of points in ray)
            - k space trajectory of gas and dissolved acquisitions (for standard 1 pt Dixon
                these are the same)
    """
    # get the raw FIDs, contrast labels, and bonus spectra labels
    raw_fids = []
    raw_traj = []
    bonus_spectra_fids = []

    contrast_labels = []
    bs_contrast_labels = []

    set_labels = []
    set_included = True
    n_projections = dataset.number_of_acquisitions()

    for i in range(0, int(n_projections)):
        acquisition_header = dataset.read_acquisition(i).getHead()

        bonus_spectra_flag = acquisition_header.measurement_uid
        if bonus_spectra_flag:
            bonus_spectra_fids.append(dataset.read_acquisition(i).data[0].flatten())
            bs_contrast_labels.append(acquisition_header.idx.contrast)

        else:

            raw_fids.append(dataset.read_acquisition(i).data[0].flatten())
            contrast_labels.append(acquisition_header.idx.contrast)
            raw_traj.append(dataset.read_acquisition(i).traj)
            try:
                set_labels.append(acquisition_header.idx.set)
            except:
                set_included = False

    bonus_spectra_fids = np.asarray(bonus_spectra_fids)
    bs_contrast_labels = np.asarray(bs_contrast_labels)

    raw_fids_truncated = np.asarray(raw_fids)
    contrast_labels_truncated = np.asarray(contrast_labels)
    set_labels_truncated = np.asarray(set_labels)
    raw_traj = np.asarray(raw_traj)
    logging.info("raw_traj_shape in get_gx_data: %s", raw_traj.shape)

    if set_included:
        unique_set_labels = np.unique(set_labels_truncated)

        gas_fids_all = []
        dis_fids_all = []
        gas_trajectories_all = []
        dis_trajectories_all = []

        for set_label in unique_set_labels:
            gas_fids_set = raw_fids_truncated[
                (contrast_labels_truncated == constants.ContrastLabels.GAS) & (
                    set_labels_truncated == set_label)]
            dis_fids_set = raw_fids_truncated[
                (contrast_labels_truncated == constants.ContrastLabels.DISSOLVED) & (
                    set_labels_truncated == set_label)]
            gas_traj_set = raw_traj[
                (contrast_labels_truncated == constants.ContrastLabels.GAS) & (
                    set_labels_truncated == set_label)]
            dis_traj_set = raw_traj[
                (contrast_labels_truncated == constants.ContrastLabels.DISSOLVED) & (
                    set_labels_truncated == set_label)]

            if gas_fids_set.size > 0 and not np.all(gas_fids_set == 0):
                gas_fids_all.append(np.expand_dims(gas_fids_set, axis=-1))
                gas_trajectories_all.append(np.expand_dims(gas_traj_set, axis=-1))
            if dis_fids_set.size > 0 and not np.all(dis_fids_set == 0):
                dis_fids_all.append(np.expand_dims(dis_fids_set, axis=-1))
                dis_trajectories_all.append(np.expand_dims(dis_traj_set, axis=-1))

        gas_fids_all = np.concatenate(gas_fids_all, axis=-1)
        dis_fids_all = np.concatenate(dis_fids_all, axis=-1)
        gas_trajectories_all = np.concatenate(gas_trajectories_all, axis=-1)
        dis_trajectories_all = np.concatenate(dis_trajectories_all, axis=-1)

        if multi_echo:
            all_traj = [gas_trajectories_all , dis_trajectories_all]
            return {
                constants.IOFields.FIDS: raw_fids_truncated,
                constants.IOFields.FIDS_GAS: gas_fids_all,
                constants.IOFields.FIDS_DIS: dis_fids_all,
                constants.IOFields.TRAJ: all_traj,
            }
        else:
            all_traj = [gas_trajectories_all[...,0] , dis_trajectories_all[...,0]]
            return {
                constants.IOFields.FIDS: raw_fids_truncated,
                constants.IOFields.FIDS_GAS: gas_fids_all[...,0],
                constants.IOFields.FIDS_DIS: dis_fids_all[...,0],
                constants.IOFields.TRAJ: all_traj,
            }

    else:
        gas_traj = raw_traj[
                contrast_labels_truncated == constants.ContrastLabels.GAS, :, :
            ]

        dis_traj = raw_traj[
                contrast_labels_truncated == constants.ContrastLabels.DISSOLVED, :, :
            ]

        all_traj = [gas_traj , dis_traj]

        return {
            constants.IOFields.FIDS: raw_fids_truncated,
            constants.IOFields.FIDS_GAS: raw_fids_truncated[
                contrast_labels_truncated == constants.ContrastLabels.GAS, :
            ],
            constants.IOFields.FIDS_DIS: raw_fids_truncated[
                contrast_labels_truncated == constants.ContrastLabels.DISSOLVED, :
            ],
            constants.IOFields.TRAJ: all_traj,
        }

# def mrd_headerRead(header):
#     """This function will take mrd header object and create header dictionary
#     for the report and calculation"""

#     mrd_headerDict = {}

#     # subject information

#     # patient information

#     # acquisition system information
#     mrd_headerDict["system"] = header.acquisitionSystemInformation.systemVendor
#     mrd_headerDict[
#         "mag_strength"
#     ] = header.acquisitionSystemInformation.systemFieldStrength_T
#     mrd_headerDict["ins_name"] = header.acquisitionSystemInformation.institutionName

#     if mrd_headerDict["ins_name"] == None:
#         mrd_headerDict["ins_name"] = "Cincinnati"

#     # encoding
#     enc = header.encoding[0]
#     mrd_headerDict[
#         "matrixSize"
#     ] = enc.reconSpace.matrixSize.z  # image should be reconstructed at this
#     mrd_headerDict["FOV"] = int(enc.reconSpace.fieldOfView_mm.x / 10.0)  # 40

#     try:
#         mrd_headerDict["dwell_time"] = enc.trajectoryDescription.userParameterDouble[
#             0
#         ].value  # 10
#     except:
#         mrd_headerDict["dwell_time"] = 20

#     # Reading/converting RF excitation
#     try:
#         gasExciFreq = enc.trajectoryDescription.userParameterDouble[2].value  # in Hz
#         disExciFreq = enc.trajectoryDescription.userParameterDouble[3].value  # in Hz
#         excitation = disExciFreq - gasExciFreq

#         gyro_ratio = 11.777  # gyromagnetic ratio of 129Xe in MHz/Tesla
#         mrd_headerDict["RF_excitation"] = round(
#             excitation / (gyro_ratio * mrd_headerDict["mag_strength"]), 1
#         )
#         #logging.info("@@@@@@@@@@@@@@@@@@")
#         #logging.info(gasExciFreq )
#     except:
#         try:
#             mrd_headerDict["RF_excitation"] = round(
#                 header.userParameters.userParameterDouble[0].value
#             )
#         except:
#             mrd_headerDict["RF_excitation"] = 0

#     # Institution specific paramters >>>>>
#     if mrd_headerDict["ins_name"] == "University of Iowa":
#         mrd_headerDict["TE90"] = np.round(
#             header.sequenceParameters.TE[0] * 1000 * 1000, 2
#         )  # 458 here # 500 for Duke
#         mrd_headerDict["tr_dis"] = np.round(
#             2 * header.sequenceParameters.TR[0] * 1000
#         )  # 15; mrd header -> 0.0075
#         mrd_headerDict["gasFA"] = 0.5  # not sure about this
#         mrd_headerDict["disFA"] = header.sequenceParameters.flipAngle_deg[1]  # 20
#         #mrd_headerDict["scan_date"] = header.studyInformation.studyDate
#         scan_date = header.measurementInformation.frameOfReferenceUID
#         mrd_headerDict["scan_date"] = (
#             "Sup" + "-" + "Date" + "-" + "NorealDate"
#         )  # YYYY-MM-DD

#     elif mrd_headerDict["ins_name"] == "St. Joseph's Healthcare Hamilton":  # Mcmaster
#         mrd_headerDict[
#             "TE90"
#         ] = 450  # np.round(header.sequenceParameters.TE[0] * 1000, 2) # 470.6; 500 for Duke
#         mrd_headerDict["tr_dis"] = np.round(
#             2 * header.sequenceParameters.TR[0] * 1000
#         )  # TR is 4.24
#         mrd_headerDict["gasFA"] = 0.5  # not sure about this
#         mrd_headerDict["disFA"] = header.sequenceParameters.flipAngle_deg[0]  # 20
#         mrd_headerDict["scan_date"] = header.studyInformation.studyDate

#     elif (
#         mrd_headerDict["ins_name"] == "Cincinnati"
#         or mrd_headerDict["ins_name"] == "CCHMC"
#     ):
#         mrd_headerDict["TE90"] = (
#             header.sequenceParameters.TE[0] * 1000
#         )  # 470.6 ; 500 for Duke
#         mrd_headerDict["tr_dis"] = 2 * header.sequenceParameters.TR[0]  # TR is 4.24
#         mrd_headerDict["gasFA"] = header.sequenceParameters.flipAngle_deg[0]  # 0.5
#         mrd_headerDict["disFA"] = header.sequenceParameters.flipAngle_deg[1]  # 15
#         scan_date = header.measurementInformation.frameOfReferenceUID
#         mrd_headerDict["scan_date"] = (
#             scan_date[:4] + "-" + scan_date[4:6] + "-" + scan_date[6:]
#         )  # YYYY-MM-DD

#     else:
#         scan_date = header.measurementInformation.frameOfReferenceUID
#         mrd_headerDict["scan_date"] = (
#             scan_date[:4] + "-" + scan_date[4:6] + "-" + scan_date[6:]
#         )  # YYYY-MM-DD
#         mrd_headerDict["gasFA"] = header.sequenceParameters.flipAngle_deg[0]  # 0.5
#         mrd_headerDict["disFA"] = header.sequenceParameters.flipAngle_deg[1]  # 15

#     # # Siemens software version - not required for other vendors/institution
#     # try:
#     #     mrd_headerDict["software_version"] = twix_obj.hdr.Dicom.SoftwareVersions
#     # except:
#     #     mrd_headerDict["software_version"] = "NA"

#     return mrd_headerDict

# def get_gx_data(dataset: ismrmrd.hdf5.Dataset) -> Dict[str, Any]:
#     """Get the FID acquisition data from dixon MRD file.
#     Args:
#         dataset: ismrmrd dataset object
#     Returns:
#         a dictionary containing
#             - all raw fids of shape (number of projections for gas and dissolved phase combined,
#                 number of points in ray)
#             - gas phase fids in shape (number of projections, number of points in ray)
#             - dissolved phase fids in shape (number fo projections, number of points in ray)
#             - k space trajectory of gas and dissolved acquisitions (for standard 1 pt Dixon
#                 these are the same)
#     """
#     # Reading the dataset
#     dset = dataset
#     nFids = dset.number_of_acquisitions()

#     # Getting Information from the Header associated with the scan
#     header = ismrmrd.xsd.CreateFromDocument(dset.read_xml_header())
#     mrd_headerDict = mrd_headerRead(header)
# #acquisition_header = dataset.read_acquisition(i).getHead()
#     # Prepare the K-space and trajectories >>>>>>>>>>>>>>
#     ## ======================= Reshaping K-space =============================
#     k_space_reshaped_all = []
#     nFrames = dset.number_of_acquisitions()
#     npts_fid = dset.read_acquisition(0).data[0].shape[0]  # 64 for cchmc

#     for i in range(0, nFrames):
#         k_space_reshaped = dset.read_acquisition(i).data[0].reshape(1, npts_fid)
#         k_space_reshaped_all.append(k_space_reshaped)

#     data_dixon = np.concatenate(k_space_reshaped_all, axis=0)

#     # Separating Gas and Dissolved Fids
#     data_gas = data_dixon[0::2, :]
#     nFrames_gas_all = data_gas.shape[0]

#     data_dis_all = data_dixon[1::2, :]
#     data_dis = data_dis_all
#     ## ===================== Reshaping trajectories ===========================
#     traj_list_all = []
#     traj_all = np.empty((nFrames, npts_fid, 3))

#     for i in range(0, nFrames):
#         traj_all[i, :, :] = dset.read_acquisition(i).traj

#     # # Separate Gas and Dissolved Trajectories - here gas and dissolved trajs are same
#     traj_gas = traj_all[0::2, :].astype(np.float64)
#     traj_dis = traj_all[1::2, :].astype(np.float64)

#     return {
#         constants.IOFields.FIDS: data_dixon,
#         constants.IOFields.FIDS_GAS: data_gas,
#         constants.IOFields.FIDS_DIS: data_dis,
#         constants.IOFields.TRAJ: [traj_gas,traj_dis],
#         "matrix_size": mrd_headerDict["matrixSize"]
#     }

def get_ute_data(dataset: ismrmrd.hdf5.Dataset) -> Dict[str, Any]:
    """Get the FID acquisition data from proton MRD file.

    Args:
        dataset: ismrmrd dataset object
    Returns:
        a dictionary containing
            - all proton fids of shape (number of projections, number of points in ray)
            - k space trajectory of proton acquisitions
    """
    # get the raw FIDs, contrast labels, and bonus spectra labels
    raw_fids = []
    contrast_labels = []
    bonus_spectra_labels = []
    n_projections = dataset.number_of_acquisitions()
    for i in range(0, int(n_projections)):
        acquisition_header = dataset.read_acquisition(i).getHead()
        raw_fids.append(dataset.read_acquisition(i).data[0].flatten())
        contrast_labels.append(acquisition_header.idx.contrast)
        bonus_spectra_labels.append(acquisition_header.measurement_uid)
    raw_fids = np.asarray(raw_fids)
    contrast_labels = np.asarray(contrast_labels)
    bonus_spectra_labels = np.asarray(bonus_spectra_labels)

    # remove bonus spectra
    raw_fids_truncated = raw_fids[
        bonus_spectra_labels == constants.BonusSpectraLabels.NOT_BONUS, :
    ]
    contrast_labels_truncated = contrast_labels[
        bonus_spectra_labels == constants.BonusSpectraLabels.NOT_BONUS
    ]

    # get the trajectories
    raw_traj = np.empty((raw_fids_truncated.shape[0], raw_fids_truncated.shape[1], 3))
    for i in range(0, raw_fids_truncated.shape[0]):
        raw_traj[i, :, :] = dataset.read_acquisition(i).traj

    return {
        constants.IOFields.FIDS: raw_fids_truncated[
            contrast_labels_truncated == constants.ContrastLabels.PROTON, :
        ],
        constants.IOFields.TRAJ: raw_traj[
            contrast_labels_truncated == constants.ContrastLabels.PROTON, :, :
        ],
    }

# def get_dwell_time(header: ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader) -> float:
#     """Get the dwell time from the MRD header.

#     Args:
#         header (ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader): MRD header
#     Returns:
#         float: dwell time in seconds
#     """
#     return 1e-6 * header.encoding[0].trajectoryDescription.userParameterDouble[0].value

# def get_excitation_freq(
#     header: ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader,
# ) -> float:
#     """Get the excitation frequency from the MRD header.

#     Args:
#         header (ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader): MRD header

#     Returns:
#         float: excitation frequency in MHz
#     """
#     return header.encoding[0].trajectoryDescription.userParameterDouble[1].value

# def get_center_freq(header: ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader) -> float:
#     """Get the center frequency from the MRD header.

#     See: https://mriquestions.com/center-frequency.html for definition of center freq.
#     Args:
#         header (ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader): MRD header

#     Returns:
#         float: center frequency in MHz
#     """
#     return 1e-6 * float(header.userParameters.userParameterLong[0].value)

# def get_excitation_freq(
#     header: ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader,
# ) -> float:
#     """Get the excitation frequency from the MRD header.

#     See: https://mriquestions.com/center-frequency.html
#     Args:
#         header (ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader): MRD header

#     Returns:
#         float: excitation frequency in ppm
#     """
#     return header.userParameters.userParameterDouble[0].value

# def get_ramp_time(header: ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader) -> float:
#     """Get the ramp time in micro-seconds.

#     See: https://mriquestions.com/gradient-specifications.html

#     Args:
#         header (ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader): MRD header
#     Returns:
#         ramp time in us
#     """
#     ramp_time = 0.0
#     try:
#         ramp_time = float(
#             header.encoding[0].trajectoryDescription.userParameterLong[0].value
#         )
#     except:
#         pass

#     return max(100, ramp_time) if ramp_time < 100 else ramp_time

# def get_TR_dissolved(header: ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader) -> float:
#     """Get the TR in seconds for dissolved phase.

#     The dissolved phase TR is defined to be the time between two consecutive dissolved
#     phase-FIDS. This is different from the TR in the mrd header as the mrd header
#     provides the TR for two consecutive FIDS. Here, we assume an interleaved sequence.

#     Args:
#         header (ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader): MRD header
#     Returns:
#         TR in seconds
#     """
#     try:
#         return 2 * header.sequenceParameters.TR[0] * 1e-3
#     except:
#         pass

#     raise ValueError("Could not find TR from twix object")

# def get_gx_data(
#     dataset: ismrmrd.hdf5.Dataset,
#     header: ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader,
# ) -> dict[str, Any]:
#     """Get the dissolved phase and gas phase FIDs from twix object.

#     For reconstruction, we also need important information like the gradient delay,
#     number of fids in each phase, etc. Note, this cannot be trivially read from the
#     twix object, and need to hard code some values. For example, the gradient delay
#     is slightly different depending on the scanner.
#     Args:
#         twix_obj: twix object returned from mapVBVD function
#     Returns:
#         a dictionary containing
#         1. dissolved phase FIDs in shape (number of projections,
#             number of points in ray).
#         2. gas phase FIDs in shape (number of projections, number of points in ray).
#         3. trajectory in shape (number of projections, number of points in ray, 3).
#             assumed that the trajectory is the same for both phases.
#         3. number of fids in each phase, used for trajectory calculation. Note:
#             this may not always be equal to the shape in 1 and 2.
#         4. number of FIDs to skip from the beginning. This may be due to a noise frame.
#         5. number of FIDs to skip from the end. This may be due to calibration.
#         6. gradient delay x in microseconds.
#         7. gradient delay y in microseconds.
#         8. gradient delay z in microseconds.
#     """
#     institution = get_institution_name(header)
#     # get the raw FIDs
#     raw_fids = []
#     n_projections = dataset.number_of_acquisitions()
#     for i in range(0, int(n_projections)):  # type: ignore
#         raw_fids.append(dataset.read_acquisition(i).data[0].flatten())
#     raw_fids = np.asarray(raw_fids)
#     # get the trajectories
#     raw_traj = np.empty((raw_fids.shape[0], raw_fids.shape[1], 3))
#     for i in range(0, int(n_projections)):  # type: ignore
#         raw_traj[i, :, :] = dataset.read_acquisition(i).traj

#     if institution == "CCHMC" and raw_fids.shape[1] == 128:
#         raw_traj = 0.5 * raw_traj
#     return {
#         constants.IOFields.FIDS_GAS: raw_fids[0::2, :],
#         constants.IOFields.FIDS_DIS: raw_fids[1::2, :],
#         constants.IOFields.TRAJ: raw_traj[0::2, :, :],
#         constants.IOFields.N_FRAMES: raw_fids.shape[0] // 2,
#         constants.IOFields.N_SKIP_START: 0,
#         constants.IOFields.N_SKIP_END: 0,
#         constants.IOFields.GRAD_DELAY_X: 0,
#         constants.IOFields.GRAD_DELAY_Y: 0,
#         constants.IOFields.GRAD_DELAY_Z: 0,
#     }
