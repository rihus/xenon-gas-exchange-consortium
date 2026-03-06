"""Module for oscillation imaging subject."""

import glob
import logging
import os
from typing import Any, Dict

import nibabel as nib
import numpy as np

import biasfield
import oscillation_binning as ob
import preprocessing as pp
import reconstruction
import registration
import segmentation
from config import base_config
from utils import (
    binning,
    constants,
    img_utils,
    io_utils,
    metrics,
    plot,
    recon_utils,
    report,
    signal_utils,
    spect_utils,
    traj_utils,
)


class Subject(object):
    """Module to for processing oscillation imaging.

    Attributes:
        config (config_dict.ConfigDict): config dict
        data_dissolved (np.array): dissolved-phase data of shape
            (n_projections, n_points)
        data_dis_high (np.array): high-key dissolved-phase data of shape
            (n_projections, n_points)
        data_dis_low (np.array): low-key dissolved-phase data of shape
            (n_projections, n_points)
        data_gas (np.array): gas-phase data of shape (n_projections, n_points)
        data_ute (np.array): UTE proton data of shape (n_projections, n_points)
        dict_bonus (dict): dictionary of bonus data and metadata
        dict_dis (dict): dictionary of dissolved-phase data and metadata
        dict_dyn (dict): dictionary of dynamic spectroscopy data and metadata
        dict_ute (dict): dictionary of UTE proton data and metadata
        high_indices (np.array): indices of high projections of shape (n, )
        low_indices (np.array): indices of low projections of shape (n, )
        image_dissolved (np.array): dissolved-phase image
        image_dissolved_norm (np.array): dissolved-phase image reconstructed with
            the data normalized by gas-phase k0
        image_gas (np.array): gas-phase image
        image_membrane (np.array): membrane image
        image_membrane2gas (np.array): membrane image normalized by gas-phase image
        image_rbc (np.array): RBC image
        image_rbc_norm (np.array): RBC image normalized of image_dissolved_norm
        image_rbc2gas (np.array): RBC image normalized by gas-phase image
        image_rbc_high (np.array): RBC image reconstructed with high-key data
        image_rbc_low (np.array): RBC image reconstructed with low-key data
        image_rbc_osc (np.array): RBC oscillation amplitude image
        image_rbc_osc_binned (np.array): RBC oscillation amplitude image binned
        image_ute (np.array): UTE proton image
        key_radius (int): radius of the keyhole in points
        low_indices (np.array): indices of low projections of shape (n, )
        mask (np.array): thoracic cavity mask
        mask_rbc (np.array): thoracic cavity mask with low SNR RBC voxels removed
        rbc_m_ratio (float): RBC to M ratio
        rbc_m_ratio_high (float): RBC to M ratio of high-key data
        rbc_m_ratio_low (float): RBC to M ratio of low-key data
        stats_dict (dict): dictionary of statistics
        traj_dissolved (np.array): dissolved-phase trajectory of shape
            (n_projections, n_points, 3)
        traj_gas (np.array): gas-phase trajectory of shape
            (n_projections, n_points, 3)
        traj_ute (np.array): UTE proton trajectory of shape
    """

    def __init__(self, config: base_config.Config):
        """Init object."""
        logging.info("Initializing gas-exchange and oscillation imaging subject.")
        self.config = config
        self.correction_map = np.array([0.0])
        self.data_dissolved = np.array([])
        self.data_dissolved_norm = np.array([])
        self.data_gas = np.array([])
        self.data_rbc_k0 = np.array([])
        self.dict_dis = {}
        self.dict_dyn = {}
        self.dict_ute = {}
        self.dict_bonus = {}
        self.dict_stats = {}
        self.dict_info = {}
        self.high_indices = np.array([0.0])
        self.image_dissolved = np.array([0.0])
        self.image_dissolved_norm = np.array([0.0])
        self.image_dissolved_high = np.array([0.0])
        self.image_dissolved_low = np.array([0.0])
        self.image_biasfield = np.array([0.0])
        self.image_gas_binned = np.array([0.0])
        self.image_gas_cor = np.array([0.0])
        self.image_gas_highsnr = np.array([0.0])
        self.image_gas_highreso = np.array([0.0])
        self.image_gas_cs = np.array([0.0])
        self.image_membrane = np.array([0.0])
        self.image_membrane2gas = np.array([0.0])
        self.image_membrane2gas_binned = np.array([0.0])
        self.image_proton = np.array([0.0])
        self.image_proton_reg = np.array([0.0])
        self.image_rbc = np.array([0.0])
        self.image_rbc_norm = np.array([0.0])
        self.image_rbc2gas = np.array([0.0])
        self.image_rbc2gas_binned = np.array([0.0])
        self.image_rbc_high = np.array([0.0])
        self.image_rbc_low = np.array([0.0])
        self.image_rbc_osc = np.array([0.0])
        self.image_rbc_osc_binned = np.array([0.0])
        self.mask_vent = np.array([0.0])
        self.key_radius = 0
        self.low_indices = np.array([0.0])
        self.mask = np.array([0.0])
        self.mask_rbc = np.array([0.0])
        self.rbc_m_ratio = 0.0
        self.rbc_m_ratio_high = 0.0
        self.rbc_m_ratio_low = 0.0
        self.relative_vc_map = np.array([0.0])
        self.traj_scaling_factor = 1.0
        self.traj_dissolved = np.array([])
        self.traj_gas = np.array([])
        self.traj_ute = np.array([])
        self.data_ute = np.array([])
        self.reference_data_key = str()
        self.reference_data = {}
        self.rbc_hb_correction_factor = "NA"
        self.membrane_hb_correction_factor = "NA"
        self.vol_correction_factor_rbc = "NA"
        self.vol_correction_factor_membrane = "NA"
        self.corrected_lung_volume = "NA"
        self.predicted_volume = "NA"
        self.extent_fac = 9

    def read_twix_files(self):
        """Read in twix files to dictionary.

        Read in the dynamic spectroscopy (if it exists) and the dissolved-phase image
        data.
        """
        self.dict_dis = io_utils.read_dis_twix(
            io_utils.get_dis_twix_files(str(self.config.data_dir))
        )
        try:
            self.dict_dyn = io_utils.read_dyn_twix(
                io_utils.get_dyn_twix_files(str(self.config.data_dir))
            )
        except ValueError:
            logging.info("No dynamic spectroscopy twix file found")
        if self.config.remove_contamination:
            self.dict_bonus = io_utils.read_bonus_twix(
                io_utils.get_dis_twix_files(str(self.config.data_dir))
            )
        if self.config.recon.recon_proton:
            self.dict_ute = io_utils.read_ute_twix(
                io_utils.get_ute_twix_files(str(self.config.data_dir))
            )

    def read_mrd_files(self):
        """Read in mrd files to dictionary.

        Read in the dynamic spectroscopy (if it exists) and the dissolved-phase image
        data.
        """
        self.dict_dis = io_utils.read_dis_mrd(
            io_utils.get_dis_mrd_files(str(self.config.data_dir))
            )# , self.config.multi_echo
        logging.info("TR read from MRD: %s", self.dict_dis[constants.IOFields.TR])
        try:
            self.dict_dyn = io_utils.read_dyn_mrd(
                io_utils.get_dyn_mrd_files(str(self.config.data_dir))
            )
        except ValueError:
            logging.info("No dynamic spectroscopy MRD file found")
        ##Recon proton if True (default: False, can be overwritten in subject config)
        if self.config.recon.recon_proton:
            self.dict_ute = io_utils.read_ute_mrd(
                io_utils.get_ute_mrd_files(str(self.config.data_dir))
            )

    def read_dicom_files(self):
        """Read in DICOM files for proton image."""
        self.image_proton = io_utils.read_dicom(
            self.config.dicom_proton_dir, self.image_gas_highreso.shape
        )

    def read_mat_file(self):
        """Read in mat file of reconstructed images.

        Note: The mat file variable names are matched to the instance variable names.
        Thus, if the variable names are changed in the mat file, they must be changed.
        """
        mdict = io_utils.import_mat(io_utils.get_mat_file(str(self.config.data_dir)))
        self.dict_dis = io_utils.import_matstruct_to_dict(mdict["dict_dis"])
        if "dict_dyn" in mdict.keys() and mdict["dict_dyn"].flatten()[0] is not None:
            logging.info("Dynamic data found.")
            self.dict_dyn = io_utils.import_matstruct_to_dict(mdict["dict_dyn"])
        if "dict_bonus" in mdict.keys() and mdict["dict_bonus"].flatten()[0] is not None:
            logging.info("Bonus data found.")
            self.dict_bonus = io_utils.import_matstruct_to_dict(mdict["dict_bonus"])
        if "dict_ute" in mdict.keys() and mdict["dict_ute"].flatten()[0] is not None:
            logging.info("UTE proton data found.")
            self.dict_ute = io_utils.import_matstruct_to_dict(mdict["dict_ute"])
        self.data_dissolved = mdict["data_dissolved"]
        self.data_dissolved_norm = mdict["data_dissolved_norm"]
        self.data_gas = mdict["data_gas"]
        self.data_rbc_k0 = mdict["data_rbc_k0"].flatten()
        self.high_indices = mdict["high_indices"].flatten()
        self.image_dissolved = mdict["image_dissolved"]
        self.image_dissolved_high = mdict["image_dissolved_high"]
        self.image_dissolved_low = mdict["image_dissolved_low"]
        self.image_dissolved_norm = mdict["image_dissolved_norm"]
        self.image_gas_highsnr = mdict["image_gas_highsnr"]
        self.key_radius = int(mdict["key_radius"])
        self.low_indices = mdict["low_indices"].flatten()
        self.mask = mdict["mask"].astype(bool)
        self.rbc_m_ratio = float(mdict["rbc_m_ratio"].flatten()[0])
        self.rbc_m_ratio_high = float(mdict["rbc_m_ratio_high"].flatten()[0])
        self.rbc_m_ratio_low = float(mdict["rbc_m_ratio_low"].flatten()[0])
        self.traj_dissolved = mdict["traj_dissolved"]
        self.traj_gas = mdict["traj_gas"]

    def calculate_rbc_m_ratio(self):
        """Calculate RBC:M ratio using static spectroscopy.

        If a manual RBC:M ratio is specified, use that instead.
        """
        if self.config.rbc_m_ratio > 0:  # type: ignore
            self.rbc_m_ratio = float(self.config.rbc_m_ratio)  # type: ignore
            logging.info("Using user-provided RBC:M ratio of %s", self.rbc_m_ratio)
        else:
            logging.info("Calculating RBC:M ratio from static spectroscopy.")
            assert self.dict_dyn[constants.IOFields.FIDS_DIS] is not None
            self.rbc_m_ratio, _ = spect_utils.calculate_static_spectroscopy(
                fid=self.dict_dyn[constants.IOFields.FIDS_DIS],
                sample_time=self.dict_dyn[constants.IOFields.SAMPLE_TIME],
                tr=self.dict_dyn[constants.IOFields.TR],
                center_freq=self.dict_dyn[constants.IOFields.XE_CENTER_FREQUENCY],
                rf_excitation=self.dict_dyn[constants.IOFields.XE_DISSOLVED_OFFSET_FREQUENCY],
                plot=False,
            )


    def preprocess(self):
        """Prepare data and trajectory for reconstruction.

        NOTE: for standard 1pt Dixon sequence, gas and dissolved trajectories are the same.

        Also, calculates the scaling factor for the trajectory.
        """
        # remove contamination
        if self.config.recon.remove_contamination:
            self.dict_dis = pp.remove_contamination(self.dict_dyn, self.dict_dis)

        self.data_dissolved = self.dict_dis[constants.IOFields.FIDS_DIS]
        self.data_gas = self.dict_dis[constants.IOFields.FIDS_GAS]

        if (self.dict_dis[constants.IOFields.INSTITUTION]
            == constants.Institution.IOWA.value):
            self.data_dissolved =  np.conjugate(self.data_dissolved)
            self.data_gas = np.conjugate(self.data_gas)

        # get or generate trajectories and trajectory scaling factors
        if constants.IOFields.TRAJ not in self.dict_dis.keys():
            self.traj_dissolved = pp.prepare_traj(self.dict_dis, config=self.config)
            self.traj_scaling_factor = traj_utils.get_scaling_factor(
            recon_size=int(self.config.recon.recon_size),
            n_points=self.data_gas.shape[1]
            )
            self.traj_gas = self.traj_dissolved
        else:
            self.traj_gas = self.dict_dis[constants.IOFields.TRAJ][0]
            self.traj_dissolved = self.dict_dis[constants.IOFields.TRAJ][1]
        ##For Duke protocol, CCHMC requires scaling factor and at times a trajectory file
        if self.config.recon.traj_type == constants.TrajType.HALTONSPIRAL:
            if self.config.institution == constants.Institution.CCHMC.value:
                ##CCHMC requires a unique scaling factor of 0.903
                logging.info("Applying CCHMC traj_scalling_factor: %s", 0.903)
                self.traj_scaling_factor = 0.903
            ##Loading trajectories from fixed location (case for CCHMC, post R59 upgrade)
            path = "./recon/fixed_traj/traj_before_rescaling.npy"
            if self.config.recon.philips_software == constants.PhilipsVersion.POST_R59.value:
                new_traj_bf = np.load(path)
                self.traj_gas= new_traj_bf.copy()
                self.traj_dissolved= new_traj_bf.copy()
                logging.info("######## Reconstructing with fixed trajectories ########")

        if self.config.recon.remove_noisy_projections:
            ##Using remove_noisy_projections() function, gas and dissolved data/traj don't match
            # remove noisy FIDs
            (self.data_gas, self.traj_gas, self.data_dissolved,
             self.traj_dissolved) = pp.remove_noisy_projections_interleaved(
                    self.data_gas, self.traj_gas, self.data_dissolved, self.traj_dissolved)
            logging.info("-----> remove_noisy_projections is TRUE")

        logging.info("####### Gas/Diss data/trajectories size pre data truncated #########")
        logging.info(self.data_gas.shape)
        logging.info(self.traj_gas.shape)
        logging.info(self.data_dissolved.shape)
        logging.info(self.traj_dissolved.shape)
        # """Calculate the number of frames to skip at the beginning of scan:
        #   if the prep_pulse = 'true', there is no skip frames n_skip_start=0;
        #   else calculated by dissolved flip angle"""
        # if (self.dict_dis[constants.IOFields.PREP_PULSES]
        #       == constants.PrepPulses.PREP_PULSES.value):
        #     self.config.recon.n_skip_start = 0
        #     logging.info(f"get prep_pulses:value={self.dict_dis[constants.IOFields.PREP_PULSES]}")
        # else:
        #     # Calculate the number of frames to skip at the beginning by dissolved flip angle
        #     if np.isnan(self.config.recon.n_skip_start):
        #         self.config.recon.n_skip_start = recon_utils.skip_from_flipangle(
        #                                           self.dict_dis[constants.IOFields.FA_DIS])

        # truncate gas and dissolved data and trajectories
        self.data_dissolved, self.traj_dissolved = pp.truncate_data_and_traj(
            self.data_dissolved,
            self.traj_dissolved,
            n_skip_start=int(self.config.recon.n_skip_start),
            n_skip_end=int(self.config.recon.n_skip_end),
        )
        self.data_gas, self.traj_gas = pp.truncate_data_and_traj(
            self.data_gas,
            self.traj_gas,
            n_skip_start=int(self.config.recon.n_skip_start),
            n_skip_end=int(self.config.recon.n_skip_end),
        )
        logging.info("####### Gas/Diss data/trajectories size post data truncated #########")
        logging.info(self.data_gas.shape)
        logging.info(self.traj_gas.shape)
        logging.info(self.data_dissolved.shape)
        logging.info(self.traj_dissolved.shape)
        logging.info("Trajectory scalling factor:%s", self.traj_scaling_factor)
        # rescale trajectories
        self.traj_dissolved *= self.traj_scaling_factor
        self.traj_gas *= self.traj_scaling_factor

        # prepare proton data and trajectories
        if self.config.recon.recon_proton:
            if getattr(self, "dict_ute", None):
                # get or generate trajectories
                if constants.IOFields.TRAJ not in self.dict_ute.keys():
                    self.traj_ute = pp.prepare_traj(self.dict_ute)
                else:
                    self.traj_ute = self.dict_ute[constants.IOFields.TRAJ]

                # get proton data
                self.data_ute = self.dict_ute[constants.IOFields.FIDS]

                # remove noisy FIDs
                if self.config.recon.remove_noisy_projections:
                    self.data_ute, self.traj_ute = pp.remove_noisy_projections(
                        self.data_ute, self.traj_ute
                    )

                # rescale trajectories
                self.traj_ute *= self.traj_scaling_factor
            else:
                logging.info("No dict_ute")
        # Choose appropriate reference distribution
        self.reference_data_key = self.config.reference_data_key

        if self.reference_data_key == constants.ReferenceDataKey.DUKE_REFERENCE.value:
            # Choose between 208 ppmm and 218 ppm.
            # Default to 218 if other or no value for excitation found.
            if 216 <= self.dict_dis[
                    constants.IOFields.XE_DISSOLVED_OFFSET_FREQUENCY
                ] <= 220:
                logging.info("Using Duke 218_PPM reference")
                self.reference_data = constants.ReferenceDistribution.REFERENCE_218_PPM

            elif 206 <= self.dict_dis[
                    constants.IOFields.XE_DISSOLVED_OFFSET_FREQUENCY
                ] <= 210:
                logging.info("Using 208_PPM reference")
                self.reference_data = constants.ReferenceDistribution.REFERENCE_208_PPM

            else:
                logging.info("Warning: Unrecognized excitation freq, using 218_PPM ref")
                self.reference_data = constants.ReferenceDistribution.REFERENCE_218_PPM

        elif self.reference_data_key == constants.ReferenceDataKey.MANUAL_REFERENCE.value:
            logging.info("Using REFERENCE_MANUAL...")
            self.reference_data = constants.ReferenceDistribution.REFERENCE_MANUAL


    def reconstruction_gas(self):
        """Reconstruct the gas phase image."""
        orientation = self.dict_dis[constants.IOFields.ORIENTATION]
        system_vendor = self.dict_dis[constants.IOFields.SYSTEM_VENDOR]
        # Reconstruct gas image depending on the option in subject config
        if self.config.recon.recon_key == constants.ReconKey.ROBERTSON.value:
            logging.info("########Starting Gas High SNR reconstruction##########")
            self.image_gas_highsnr = reconstruction.reconstruct(
                data=(recon_utils.flatten_data(self.data_gas)),
                traj=recon_utils.flatten_traj(self.traj_gas),
                kernel_sharpness=float(self.config.recon.kernel_sharpness_lr),
                kernel_extent=self.extent_fac * float(self.config.recon.kernel_sharpness_lr),
                image_size=int(self.config.recon.recon_size),
            )
            self.image_gas_highsnr = img_utils.interp(self.image_gas_highsnr,
                self.config.recon.matrix_size // self.config.recon.recon_size,
            )
            self.image_gas_highsnr = img_utils.flip_and_rotate_image(
                self.image_gas_highsnr,
                orientation=orientation, system_vendor=system_vendor,
            )
            io_utils.export_nii(np.abs(self.image_gas_highsnr),
                                "tmp/nii/image_gas_highsnr.nii")
            logging.info("########Starting Gas Highres reconstruction##########")
            self.image_gas_highreso = reconstruction.reconstruct(
                data=(recon_utils.flatten_data(self.data_gas)),
                traj=recon_utils.flatten_traj(self.traj_gas),
                kernel_sharpness=float(self.config.recon.kernel_sharpness_hr),
                kernel_extent=self.extent_fac * float(self.config.recon.kernel_sharpness_hr),
                image_size=int(self.config.recon.recon_size),
            )
            self.image_gas_highreso = img_utils.interp(self.image_gas_highreso,
            self.config.recon.matrix_size // self.config.recon.recon_size,
            )
            self.image_gas_highreso = img_utils.flip_and_rotate_image(
                self.image_gas_highreso,
                orientation=orientation, system_vendor=system_vendor,
            )
            io_utils.export_nii(np.abs(self.image_gas_highreso),
                                "tmp/nii/image_gas_highreso.nii")
        elif (
            self.config.recon.recon_key == constants.ReconKey.PLUMMER.value
            or self.config.recon.recon_key == constants.ReconKey.PLUMMER2.value
        ):
            norm_data = np.linalg.norm(self.data_gas)
            self.data_gas /= norm_data
            decay_factor = signal_utils.calculate_decay_factor(
                self.data_gas,
                constants.T2STAR_GAS,
                self.dict_dyn[constants.IOFields.SAMPLE_TIME],
            )
            self.image_gas_cs = (
                reconstruction.reconstruct_cs(
                    data=recon_utils.flatten_data(self.data_gas),
                    traj=recon_utils.flatten_traj(self.traj_gas),
                    image_size=int(self.config.recon.recon_size),
                    overgrid_factor=1,
                    k=decay_factor,)* norm_data)
            self.data_gas *= norm_data
            self.image_gas_cs = img_utils.flip_and_rotate_image(self.image_gas_cs,
                orientation=orientation, system_vendor=system_vendor,
            )
            io_utils.export_nii(np.abs(self.image_gas_cs), "tmp/nii/image_gas_cs.nii")
        else:
            raise ValueError(f"Unknown reconstruction key: {self.config.recon.recon_key}")


    def reconstruction_dissolved(self):
        """Reconstruct the dissolved phase image."""
        orientation = self.dict_dis[constants.IOFields.ORIENTATION]
        system_vendor = self.dict_dis[constants.IOFields.SYSTEM_VENDOR]
        if self.config.recon.recon_key == constants.ReconKey.ROBERTSON.value:
            # divide the data by the gas phase k0 data.
            self.data_dissolved_norm = pp.normalize_data(
                data=self.data_dissolved, normalization=np.abs(self.data_gas[:, 0])
            )
            logging.info("########Starting Dissolved:Gas Recon##########")
            self.image_dissolved_norm = reconstruction.reconstruct(
                data=(recon_utils.flatten_data(self.data_dissolved_norm)),
                traj=recon_utils.flatten_traj(self.traj_dissolved),
                kernel_sharpness=float(self.config.recon.kernel_sharpness_lr),
                kernel_extent=self.extent_fac * float(self.config.recon.kernel_sharpness_lr),
                image_size=int(self.config.recon.recon_size),
            )
            self.image_dissolved_norm = img_utils.interp( self.image_dissolved_norm,
                        self.config.recon.matrix_size // self.config.recon.recon_size,
            )
            self.image_dissolved_norm = img_utils.flip_and_rotate_image(
                self.image_dissolved_norm,
                orientation= orientation, system_vendor= system_vendor,
            )
            logging.info("########Starting Dissolved Recon##########")
            self.image_dissolved = reconstruction.reconstruct(
                data=(recon_utils.flatten_data(self.data_dissolved)),
                traj=recon_utils.flatten_traj(self.traj_dissolved),
                kernel_sharpness=float(self.config.recon.kernel_sharpness_lr),
                kernel_extent=self.extent_fac * float(self.config.recon.kernel_sharpness_lr),
                image_size=int(self.config.recon.recon_size),
            )
            self.image_dissolved = img_utils.interp(
            self.image_dissolved,
            self.config.recon.matrix_size // self.config.recon.recon_size,
            )
            self.image_dissolved = img_utils.flip_and_rotate_image(
                self.image_dissolved,
                orientation= orientation, system_vendor= system_vendor,
            )
            io_utils.export_nii(np.abs(self.image_dissolved_norm),
                                "tmp/nii/image_dissolved_norm.nii")
            io_utils.export_nii(np.abs(self.image_dissolved),
                                "tmp/nii/image_dissolved.nii")

        elif self.config.recon.recon_key == constants.ReconKey.PLUMMER.value:
            norm_data = np.linalg.norm(self.data_dissolved)
            self.data_dissolved /= norm_data
            decay_factor = signal_utils.calculate_decay_factor(
                self.data_dissolved,
                constants.T2STAR_DISSOLVED_3T,
                self.dict_dyn[constants.IOFields.SAMPLE_TIME],
            )
            self.image_dissolved_norm = reconstruction.reconstruct_cs(
                data=recon_utils.flatten_data(self.data_dissolved),
                traj=recon_utils.flatten_traj(self.traj_dissolved),
                image_size=int(self.config.recon.recon_size),
                overgrid_factor=1,
                k=decay_factor,
            )
            self.data_dissolved *= norm_data
            self.image_dissolved_norm *= norm_data
            # self.image_dissolved = self.image_dissolved_norm

            self.image_dissolved_norm = img_utils.flip_and_rotate_image(
                self.image_dissolved_norm,
                orientation= orientation, system_vendor= system_vendor,
            )
            self.image_dissolved = img_utils.flip_and_rotate_image(
                self.image_dissolved,
                orientation= orientation, system_vendor= system_vendor,
            )
            io_utils.export_nii(
                np.abs(self.image_dissolved_norm), "tmp/nii/image_dissolved_norm.nii"
            )
            io_utils.export_nii(
                np.abs(self.image_dissolved), "tmp/nii/image_dissolved.nii"
            )

        elif self.config.recon.recon_key == constants.ReconKey.PLUMMER2.value:
            norm_data = np.linalg.norm(self.data_dissolved)
            self.data_dissolved /= norm_data
            self.image_dissolved_norm = reconstruction.reconstruct_cs(
                data=recon_utils.flatten_data(self.data_dissolved),
                traj=recon_utils.flatten_traj(self.traj_dissolved),
                image_size=int(self.config.recon.recon_size),
                overgrid_factor = 1,
                k = None,
            )
            self.data_dissolved *= norm_data
            self.image_dissolved_norm *= norm_data
            self.image_dissolved = self.image_dissolved_norm

            io_utils.export_nii(
                np.abs(self.image_dissolved), "tmp/nii/image_dissolved.nii"
            )
        else:
            raise ValueError(f"Unknown reconstruction key: {self.config.recon.recon_key}")


    def reconstruction_rbc_oscillation(self):
        """Reconstruct the RBC oscillation image."""
        orientation = self.dict_dis[constants.IOFields.ORIENTATION]
        system_vendor = self.dict_dis[constants.IOFields.SYSTEM_VENDOR]
        # logging.info("TR: %s", self.dict_dis[constants.IOFields.TR])
        logging.info("TR_dissolved: %s", self.dict_dis[constants.IOFields.TR_DIS])
        # bin rbc oscillations
        logging.info("Starting RBC Oscillations Recon")
        (
            self.data_rbc_k0,
            self.high_indices,
            self.low_indices,
            self.rbc_m_ratio_high,
            self.rbc_m_ratio_low,
        ) = ob.bin_rbc_oscillations(
            data_gas=self.data_gas,
            data_dissolved=self.data_dissolved,
            TR= self.dict_dis[constants.IOFields.TR_DIS],
            rbc_m_ratio=self.rbc_m_ratio,
        )
        # calculate the key radius (14% of the total points per projection)
        self.key_radius = np.ceil(self.data_dissolved.shape[1]*14/100)
        if self.config.recon.recon_key == constants.ReconKey.ROBERTSON.value:
            # #prepare trajectories and reconstruct high-key data
            data_dis_high, traj_dis_high = pp.prepare_data_and_traj_keyhole(
                data=self.data_dissolved_norm,
                traj=self.traj_dissolved,
                bin_indices=self.high_indices,
                key_radius=self.key_radius,
            )
            self.image_dissolved_high = reconstruction.reconstruct(
                data=data_dis_high,
                traj=traj_dis_high,
                kernel_sharpness=float(self.config.recon.kernel_sharpness_lr),
                kernel_extent=self.extent_fac * float(self.config.recon.kernel_sharpness_lr),
                image_size=int(self.config.recon.recon_size),
            )
            self.image_dissolved_high = img_utils.interp(
            self.image_dissolved_high,
            self.config.recon.matrix_size // self.config.recon.recon_size,
            )
            self.image_dissolved_high = img_utils.flip_and_rotate_image(
                self.image_dissolved_high,
                orientation= orientation, system_vendor= system_vendor,
            )
            io_utils.export_nii(
                np.abs(self.image_dissolved_high), "tmp/nii/high_key_image.nii")

            # #prepare trajectories and reconstruct low-key data
            data_dis_low, traj_dis_low = pp.prepare_data_and_traj_keyhole(
                data=self.data_dissolved_norm,
                traj=self.traj_dissolved,
                bin_indices=self.low_indices,
                key_radius=self.key_radius,
            )
            self.image_dissolved_low = reconstruction.reconstruct(
                data=data_dis_low,
                traj=traj_dis_low,
                kernel_sharpness=float(self.config.recon.kernel_sharpness_lr),
                kernel_extent=self.extent_fac * float(self.config.recon.kernel_sharpness_lr),
                image_size=int(self.config.recon.recon_size),
            )
            self.image_dissolved_low = img_utils.interp(
            self.image_dissolved_low,
            self.config.recon.matrix_size // self.config.recon.recon_size,
            )
            self.image_dissolved_low = img_utils.flip_and_rotate_image(
                self.image_dissolved_low,
                orientation= orientation, system_vendor= system_vendor,
            )
            io_utils.export_nii(
                np.abs(self.image_dissolved_low), "tmp/nii/low_key_image.nii")

        elif self.config.recon.recon_key == constants.ReconKey.PLUMMER.value:
            # prepare data and traj for reconstruction
            norm_data = np.linalg.norm(self.data_dissolved)
            self.data_dissolved /= norm_data
            (
                data_dis_high,
                traj_dis_high,
                decay_factor_high,
            ) = pp.prepare_data_and_traj_keyhole_cs(
                data=self.data_dissolved,
                traj=self.traj_dissolved,
                bin_indices=self.high_indices,
                dwell_time=self.dict_dyn[constants.IOFields.SAMPLE_TIME],
                key_radius=self.key_radius,
            )
            (
                data_dis_low,
                traj_dis_low,
                decay_factor_low,
            ) = pp.prepare_data_and_traj_keyhole_cs(
                data=self.data_dissolved,
                traj=self.traj_dissolved,
                bin_indices=self.low_indices,
                dwell_time=self.dict_dyn[constants.IOFields.SAMPLE_TIME],
                key_radius=self.key_radius,
            )
            self.image_dissolved_high = reconstruction.reconstruct_cs(
                data=data_dis_high,
                traj=traj_dis_high,
                image_size=int(self.config.recon.recon_size),
                overgrid_factor=1,
                k=decay_factor_high,
            )
            self.image_dissolved_low = reconstruction.reconstruct_cs(
                data=data_dis_low,
                traj=traj_dis_low,
                image_size=int(self.config.recon.recon_size),
                overgrid_factor=1,
                k=decay_factor_low,
            )
            self.image_dissolved_high *= norm_data
            self.image_dissolved_low *= norm_data
            # flip and rotate images
            self.image_dissolved_high = img_utils.flip_and_rotate_image(
                self.image_dissolved_high,
                orientation= orientation, system_vendor= system_vendor,
            )
            self.image_dissolved_low = img_utils.flip_and_rotate_image(
                self.image_dissolved_low,
                orientation= orientation, system_vendor= system_vendor,
            )
        elif self.config.recon.recon_key == constants.ReconKey.PLUMMER2.value:
            # prepare data and traj for reconstruction
            norm_data = np.linalg.norm(self.data_dissolved)
            self.data_dissolved /= norm_data
            (
                data_dis_high,
                traj_dis_high,
                decay_factor_high,
            ) = pp.prepare_data_and_traj_keyhole_cs(
                data=self.data_dissolved,
                traj=self.traj_dissolved,
                bin_indices=self.high_indices,
                dwell_time=self.dict_dyn[constants.IOFields.SAMPLE_TIME],
                key_radius=self.key_radius,
            )
            (
                data_dis_low,
                traj_dis_low,
                decay_factor_low,
            ) = pp.prepare_data_and_traj_keyhole_cs(
                data=self.data_dissolved,
                traj=self.traj_dissolved,
                bin_indices=self.low_indices,
                dwell_time=self.dict_dyn[constants.IOFields.SAMPLE_TIME],
                key_radius=self.key_radius,
            )
            self.image_dissolved_high = reconstruction.reconstruct_cs(
                data=data_dis_high,
                traj=traj_dis_high,
                image_size=int(self.config.recon.recon_size),
                overgrid_factor=1,
            )
            self.image_dissolved_low = reconstruction.reconstruct_cs(
                data=data_dis_low,
                traj=traj_dis_low,
                image_size=int(self.config.recon.recon_size),
                overgrid_factor=1,
            )
            self.image_dissolved_high *= norm_data
            self.image_dissolved_low *= norm_data
        else:
            raise ValueError(f"Unknown reconstruction key: {self.config.recon.recon_key}")


    def reconstruction_ute(self):
        """Reconstruct the UTE image."""
        if self.config.recon.recon_key == constants.ReconKey.ROBERTSON.value:
            logging.info("########Starting UTE reconstruction##########")
            self.image_proton = reconstruction.reconstruct(
                data=(recon_utils.flatten_data(self.data_ute)),
                traj=recon_utils.flatten_traj(self.traj_ute),
                kernel_sharpness=float(self.config.recon.kernel_sharpness_hr),
                kernel_extent=self.extent_fac * float(self.config.recon.kernel_sharpness_hr),
                image_size=int(self.config.recon.recon_size),
            )
        elif self.config.recon.recon_key == constants.ReconKey.PLUMMER.value:
            raise NotImplementedError("Plummer CS reconstruction not implemented for UTE.")
        else:
            raise ValueError(f"Unknown reconstruction key: {self.config.recon.recon_key}")
        self.image_proton = img_utils.interp(
            self.image_proton,
            self.config.recon.matrix_size // self.config.recon.recon_size,
        )
        self.image_proton = img_utils.flip_and_rotate_image(
            self.image_proton,
            orientation= self.dict_dis[constants.IOFields.ORIENTATION],
            system_vendor= self.dict_dis[constants.IOFields.SYSTEM_VENDOR],
        )
        io_utils.export_nii(np.abs(self.image_proton), "tmp/nii/image_proton.nii")


    def segmentation(self):
        """Segment the thoracic cavity."""
        if self.config.segmentation_key == constants.SegmentationKey.CNN_VENT.value:
            logging.info("Performing neural network segmenation...")
            self.mask = segmentation.predict(self.image_gas_highreso, erosion=1)
            io_utils.export_nii(self.mask.astype(np.uint8),
                                "tmp/nii/cnn_mask.nii")
        elif self.config.segmentation_key == constants.SegmentationKey.SKIP.value:
            logging.info("Segmentation skipped... using entire image")
            self.mask = np.ones_like(self.image_gas_highreso)
        elif self.config.segmentation_key == constants.SegmentationKey.THRESHOLD_VENT.value:
            logging.info("Using Gas Vent image threshold to generate mask")
            self.mask = segmentation.threshold_mask(self.image_gas_highreso, percentile=98,
                                                    morph="erode")
            io_utils.export_nii(self.mask.astype(np.uint8),
                                "tmp/nii/threshold_mask.nii")
        elif self.config.segmentation_key == constants.SegmentationKey.MANUAL_VENT.value:
            logging.info("Manual segmentation: Loading mask file specified by the user.")
            self.mask = io_utils.load_manual_segmentation(self.config.manual_seg_filepath)
        else:
            raise ValueError(f"Invalid segmentation key: {self.config.segmentation_key}")


    def registration(self):
        """Register moving image to target image.

        Uses ANTs registration to register the proton image to the xenon image.
        """
        if self.config.registration_key == constants.RegistrationKey.MASK2GAS.value:
            logging.info("Run registration algorithm, vent is fixed, mask is moving")
            self.mask, self.image_proton_reg = np.abs(
                registration.register_ants(
                    abs(self.image_gas_highreso), self.mask, self.image_proton
                )
            )
        elif self.config.registration_key == constants.RegistrationKey.PROTON2GAS.value:
            logging.info("Run registration algorithm, vent is fixed, proton is moving")
            self.image_proton_reg, mask = np.abs(
                registration.register_ants(
                    abs(self.image_gas_highreso), self.image_proton, self.mask
                )
            )
            if (
                self.config.segmentation_key
                == constants.SegmentationKey.CNN_PROTON.value
                or self.config.segmentation_key
                == constants.SegmentationKey.MANUAL_PROTON.value
            ):
                self.mask = mask
        elif self.config.registration_key == constants.RegistrationKey.MANUAL.value:
            # Load a file specified by the user
            try:
                proton_reg = glob.glob(self.config.manual_reg_filepath)[0]
                self.image_proton_reg = np.squeeze(
                    np.array(nib.load(proton_reg).get_fdata())
                )
            except ValueError:
                logging.error("Invalid proton nifti file.")
        elif self.config.registration_key == constants.RegistrationKey.SKIP.value:
            logging.info("No registration, setting registered proton to proton")
            self.image_proton_reg = self.image_proton
        else:
            raise ValueError("Invalid registration key.")

        def convert_and_threshold_mask(mask):
            """
            Check if the mask is not boolean, then apply a threshold and convert to boolean.

            Parameters:
                mask (numpy.ndarray): Input mask array.

            Returns:
                numpy.ndarray: Mask array thresholded & converted to boolean if not already.
            """
            if mask.dtype != bool:
                mask = np.where(mask > 0.5, 1, 0)  # Apply threshold: >0.5 becomes 1, otherwise 0
                mask = mask.astype(bool)          # Convert to boolean
            return mask
        self.mask = convert_and_threshold_mask(self.mask)


    def biasfield_correction(self):
        """Correct ventilation image for bias field."""
        if self.config.bias_key == constants.BiasfieldKey.SKIP.value:
            logging.info("Skipping bias field correction.")
            self.image_gas_cor = abs(self.image_gas_highreso)
            self.image_biasfield = np.ones(self.image_gas_highreso.shape)
        elif self.config.bias_key == constants.BiasfieldKey.N4ITK.value:
            logging.info("Performing N4ITK bias field correction.")
            (
                self.image_gas_cor,
                self.image_biasfield,
            ) = biasfield.correct_biasfield_n4itk(
                image=abs(self.image_gas_highreso),
                mask=self.mask.astype(bool),
            )
        else:
            raise ValueError("Invalid bias field correction key.")


    def gas_binning(self):
        """Bin gas images to colormap bins."""
        self.image_gas_binned = binning.linear_bin(
            image=img_utils.normalize(self.image_gas_cor, self.mask),
            mask=self.mask,
            thresholds=self.reference_data['threshold_vent'],
        )
        self.mask_vent = np.logical_and(self.image_gas_binned > 1, self.mask)


    def dixon_decomposition(self):
        """Perform Dixon decomposition on the dissolved-phase images."""
        rbc_m_ratio = (
            -self.rbc_m_ratio
            if self.config.recon.recon_key == constants.ReconKey.PLUMMER.value
            else self.rbc_m_ratio
        )
        rbc_m_ratio_high = (
            -self.rbc_m_ratio_high
            if self.config.recon.recon_key == constants.ReconKey.PLUMMER.value
            else self.rbc_m_ratio_high
        )
        rbc_m_ratio_low = (
            -self.rbc_m_ratio_low
            if self.config.recon.recon_key == constants.ReconKey.PLUMMER.value
            else self.rbc_m_ratio_low
        )
        logging.info("######## Gas/Dissolved/Mask Image shape #########")
        logging.info(self.image_gas_highsnr.shape)
        logging.info(self.image_dissolved.shape)
        logging.info(self.mask.shape)
        self.image_rbc, self.image_membrane = img_utils.dixon_decomposition(
            image_gas=self.image_gas_highsnr,
            image_dissolved=self.image_dissolved,
            mask=self.mask_vent,
            rbc_m_ratio=rbc_m_ratio,
        )
        self.image_rbc_high, _ = img_utils.dixon_decomposition(
            image_gas=self.image_gas_highsnr,
            image_dissolved=self.image_dissolved_high,
            mask=self.mask,
            rbc_m_ratio=rbc_m_ratio_high,
        )
        self.image_rbc_low, _ = img_utils.dixon_decomposition(
            image_gas=self.image_gas_highsnr,
            image_dissolved=self.image_dissolved_low,
            mask=self.mask,
            rbc_m_ratio=rbc_m_ratio_low,
        )
        if self.config.recon.recon_key == constants.ReconKey.PLUMMER.value:
            self.image_rbc_norm = self.image_rbc
        else:
            self.image_rbc_norm = img_utils.dixon_decomposition(
                image_gas=self.image_gas_highsnr,
                image_dissolved=self.image_dissolved_norm,
                mask=self.mask,
                rbc_m_ratio=rbc_m_ratio,
            )[0]


    def hb_correction(self):
        """Apply hemoglobin correction."""
        if self.config.hb_correction_key != constants.HbCorrectionKey.NONE.value:
            if self.config.hb > 0:
                # get hb correction scaling factors
                (self.rbc_hb_correction_factor,
                self.membrane_hb_correction_factor,
                ) = signal_utils.get_hb_correction(self.config.hb)
                logging.info("Applying hemoglobin correction to RBC and membrane signal")

                # scale dissolved phase signals by hb correction scaling factors
                self.rbc_m_ratio *= (
                    self.rbc_hb_correction_factor / self.membrane_hb_correction_factor
                )
                self.image_rbc *= self.rbc_hb_correction_factor
                self.image_membrane *= self.membrane_hb_correction_factor
            else:
                raise ValueError("Invalid hemoglobin value...")
        else:
            logging.info("Skipping hemoglobin correction...")


    def vol_correction(self):
        """
        Applying volume correction to membrane and RBC signal
        """
        self.dict_stats = {constants.StatsIOFields.INFLATION: metrics.inflation_volume(
        self.mask, self.dict_dis[constants.IOFields.FOV])}
        if self.config.vol_correction_key != constants.VolCorrectionKey.NONE.value:
            if self.dict_stats["inflation"] > 0:
                self.corrected_lung_volume = self.config.corrected_lung_volume
                #get volume correction scaling factors
                (
                    self.vol_correction_factor_rbc,
                    self.vol_correction_factor_membrane,
                    self.predicted_volume,
                ) = signal_utils.get_vol_correction(self.dict_stats["inflation"],
                                                    self.corrected_lung_volume)

                if (self.config.vol_correction_key
                    == constants.VolCorrectionKey.RBC_AND_MEMBRANE.value
                ):
                    logging.info("Membrane correction factor = %s",
                                 self.vol_correction_factor_membrane)
                    logging.info("RBC correction factor = %s", self.vol_correction_factor_rbc)

                ## scale dissolved phase signals by volume correction scaling factors
                self.rbc_m_ratio /= (
                    self.vol_correction_factor_rbc / self.vol_correction_factor_membrane
                )
                self.image_rbc /= self.vol_correction_factor_rbc
                self.image_membrane /= self.vol_correction_factor_membrane
            else:
                raise ValueError("Invalid volume value")
        else:
            logging.info("Skipping volume correction")


    def dissolved_analysis(self):
        """Calculate the dissolved-phase images relative to gas image."""
        self.image_rbc2gas = img_utils.divide_images(
            image1=self.image_rbc,
            image2=np.abs(self.image_gas_highsnr),
            mask=self.mask_vent,
        )
        self.image_membrane2gas = img_utils.divide_images(
            image1=self.image_membrane,
            image2=np.abs(self.image_gas_highsnr),
            mask=self.mask_vent,
        )
        if self.config.recon.recon_key == constants.ReconKey.ROBERTSON.value:
            # Plummer recon already includes T2* correction
            # scale by flip angle difference
            flip_angle_scale_factor = signal_utils.calculate_flipangle_correction(
                self.dict_dis[constants.IOFields.FA_GAS],
                self.dict_dis[constants.IOFields.FA_DIS],
            )
            t2star_scale_factor_rbc = signal_utils.calculate_t2star_correction(
                self.dict_dis[constants.IOFields.TE90],
                constants.T2STAR_RBC_3T,
                self.dict_dis[constants.IOFields.FIELD_STRENGTH],
            )
            t2star_scale_factor_membrane = signal_utils.calculate_t2star_correction(
                self.dict_dis[constants.IOFields.TE90],
                constants.T2STAR_MEMBRANE_3T,
                self.dict_dis[constants.IOFields.FIELD_STRENGTH],
            )
            self.image_rbc2gas = (
                flip_angle_scale_factor * t2star_scale_factor_rbc * self.image_rbc2gas
            )
            self.image_membrane2gas = (
                flip_angle_scale_factor
                * t2star_scale_factor_membrane
                * self.image_membrane2gas
            )

    def dissolved_binning(self):
        """Bin dissolved images to colormap bins."""
        self.image_rbc2gas_binned = binning.linear_bin(
            image=self.image_rbc2gas,
            mask=self.mask_vent,
            thresholds=self.reference_data["threshold_rbc"],
        )
        self.image_membrane2gas_binned = binning.linear_bin(
            image=self.image_membrane2gas,
            mask=self.mask_vent,
            thresholds=self.reference_data["threshold_membrane"],
        )


    def oscillation_analysis(self):
        """Calculate the oscillation image from the rbc high, low, and normal images."""
        # calculate the mask for the RBC image with sufficient SNR, excluding defects
        image_noise = metrics.snr(self.image_rbc, self.mask)[2]
        self.mask_rbc = np.logical_and(self.mask, self.image_rbc > image_noise)

        if self.config.correction.vc_correction:
            subject_age = self.config.correction.subject_age
            subject_sex = self.config.correction.subject_sex
            subject_height = self.config.correction.subject_height
            result = img_utils.calculate_corrected_rbc_oscillation(
                self.image_rbc_high,
                self.image_rbc_low,
                self.image_rbc_norm,
                self.image_rbc2gas,
                self.mask_rbc,
                subject_age,
                subject_sex,
                subject_height,
            )
            self.relative_vc_map = result[0]
            self.correction_map = result[1]
            self.image_rbc_osc = result[2]
        else:
            self.image_rbc_osc = img_utils.calculate_rbc_oscillation(
                self.image_rbc_high,
                self.image_rbc_low,
                self.image_rbc_norm,
                self.mask_rbc,
                method = constants.Methods.SMOOTH,
            )

    def oscillation_binning(self):
        """Bin oscillation image to colormap bins."""
        self.image_rbc_osc_binned = binning.linear_bin(
            image=self.image_rbc_osc,
            mask=self.mask,
            thresholds=self.config.params.threshold_oscillation,
        )
        # set unanalyzed voxels to -1
        self.image_rbc_osc_binned[np.logical_and(self.mask, ~self.mask_rbc)] = -1

    def get_statistics(self):
        """Calculate image statistics.
        
        Returns:
            dict_stats: Dictionary of statistics for reporting
        """
        self.dict_stats = {
            constants.IOFields.SUBJECT_ID: self.config.subject_id,
            constants.IOFields.SCAN_DATE: self.dict_dis[constants.IOFields.SCAN_DATE],
            constants.IOFields.PROCESS_DATE: metrics.process_date(),
            constants.IOFields.INSTITUTION: self.config.institution,
            constants.IOFields.SYSTEM_VENDOR: self.config.system_vendor,
            constants.IOFields.PHILIPS_VERSION: self.config.recon.philips_software,
            constants.StatsIOFields.INFLATION: metrics.inflation_volume(
                self.mask, self.dict_dis[constants.IOFields.FOV]
            ),
            constants.StatsIOFields.RBC_M_RATIO: self.rbc_m_ratio,
            #SNR
            constants.StatsIOFields.VENT_SNR: metrics.snr(
                np.abs(self.image_gas_highreso), self.mask)[1],
            constants.StatsIOFields.RBC_SNR: metrics.snr(self.image_rbc, self.mask)[0],
            constants.StatsIOFields.MEMBRANE_SNR: metrics.snr(
                self.image_membrane, self.mask)[0],
            constants.StatsIOFields.SNR_RBC_HIGH: metrics.snr(
                self.image_rbc_high, self.mask)[0],
            constants.StatsIOFields.SNR_RBC_LOW: metrics.snr(
                self.image_rbc_low, self.mask
            )[0],
            constants.StatsIOFields.SNR_DISSOLVED: metrics.snr(
                np.abs(self.image_dissolved), self.mask
            )[1],
            ##Vent Stats
            constants.StatsIOFields.VENT_DEFECT_PCT: metrics.bin_percentage(
                self.image_gas_binned, np.array([1]), self.mask
            ),
            constants.StatsIOFields.VENT_LOW_PCT: metrics.bin_percentage(
                self.image_gas_binned, np.array([2]), self.mask
            ),
            constants.StatsIOFields.VENT_HIGH_PCT: metrics.bin_percentage(
                self.image_gas_binned, np.array([5, 6]), self.mask
            ),
            constants.StatsIOFields.VENT_MEAN: metrics.mean(
                img_utils.normalize(np.abs(self.image_gas_cor), self.mask), self.mask
            ),
            constants.StatsIOFields.VENT_MEDIAN: metrics.median(
                img_utils.normalize(np.abs(self.image_gas_cor), self.mask), self.mask
            ),
            constants.StatsIOFields.VENT_STDDEV: metrics.std(
                img_utils.normalize(np.abs(self.image_gas_cor), self.mask), self.mask
            ),
            ##RBC Stats
            constants.StatsIOFields.RBC_DEFECT_PCT: metrics.bin_percentage(
                self.image_rbc2gas_binned, np.array([1]), self.mask
            ),
            constants.StatsIOFields.RBC_LOW_PCT: metrics.bin_percentage(
                self.image_rbc2gas_binned, np.array([2]), self.mask
            ),
            constants.StatsIOFields.RBC_HIGH_PCT: metrics.bin_percentage(
                self.image_rbc2gas_binned, np.array([5, 6]), self.mask
            ),
            constants.StatsIOFields.RBC_MEAN: metrics.mean(
                self.image_rbc2gas, self.mask_vent
            ),
            constants.StatsIOFields.RBC_MEDIAN: metrics.median(
                self.image_rbc2gas, self.mask_vent
            ),
            constants.StatsIOFields.RBC_STDDEV: metrics.std(
                self.image_rbc2gas, self.mask_vent
            ),
            ##Membrane Stats
            constants.StatsIOFields.MEMBRANE_DEFECT_PCT: metrics.bin_percentage(
                self.image_membrane2gas_binned, np.array([1]), self.mask
            ),
            constants.StatsIOFields.MEMBRANE_LOW_PCT: metrics.bin_percentage(
                self.image_membrane2gas_binned, np.array([2]), self.mask
            ),
            constants.StatsIOFields.MEMBRANE_HIGH_PCT: metrics.bin_percentage(
                self.image_membrane2gas_binned, np.array([6, 7, 8]), self.mask
            ),
            constants.StatsIOFields.MEMBRANE_MEAN: metrics.mean(
                self.image_membrane2gas, self.mask_vent
            ),
            constants.StatsIOFields.MEMBRANE_MEDIAN: metrics.median(
                self.image_membrane2gas, self.mask_vent
            ),
            constants.StatsIOFields.MEMBRANE_STDDEV: metrics.std(
                self.image_membrane2gas, self.mask_vent
            ),
            ##RBC Oscillations Stats
            constants.StatsIOFields.OSC_DEFECT_PCT: metrics.bin_percentage(
                self.image_rbc_osc_binned, np.array([1])
            ),
            constants.StatsIOFields.OSC_LOW_PCT: metrics.bin_percentage(
                self.image_rbc_osc_binned, np.array([2])
            ),
            constants.StatsIOFields.OSC_HIGH_PCT: metrics.bin_percentage(
                self.image_rbc_osc_binned, np.array([6, 7, 8])
            ),
            constants.StatsIOFields.OSC_MEAN: metrics.mean(
                self.image_rbc_osc, self.mask_rbc
            ),
            constants.StatsIOFields.OSC_MEDIAN: metrics.median(
                self.image_rbc_osc, self.mask_rbc
            ),
            constants.StatsIOFields.OSC_STDDEV: metrics.std(
                self.image_rbc_osc, self.mask_rbc
            ),
            constants.StatsIOFields.OSC_DEFECTLOW_PCT: metrics.bin_percentage(
                self.image_rbc_osc_binned, np.array([1, 2])
            ),
            constants.StatsIOFields.PCT_OSC_NEGATIVE: metrics.negative_voxels_percentage(
                self.image_rbc_osc, self.mask_rbc
            ),
            ##Other metrics
            constants.StatsIOFields.KEY_RADIUS: self.key_radius,
            constants.StatsIOFields.N_POINTS: self.data_gas.shape[1],
            constants.StatsIOFields.ALVEOLAR_VOLUME: metrics.alveolar_volume(
                self.image_gas_binned, self.mask, self.dict_dis[constants.IOFields.FOV]
            ),
            constants.StatsIOFields.KCO_EST: metrics.kco(
                self.image_membrane2gas,
                self.image_rbc2gas,
                self.mask_vent,
                self.reference_data['reference_fit_membrane'][1],
                self.reference_data['reference_fit_rbc'][1],
            ),
            constants.StatsIOFields.DLCO_EST: metrics.dlco(
                self.image_gas_binned,
                self.image_membrane2gas,
                self.image_rbc2gas,
                self.mask,
                self.mask_vent,
                self.dict_dis[constants.IOFields.FOV],
                self.reference_data['reference_fit_membrane'][1],
                self.reference_data['reference_fit_rbc'][1],
            ),
        }

        return self.dict_stats


    def get_info(self) -> Dict[str, Any]:
        """Gather information about the data and processing steps.

        Returns:
            dict_info: Dictionary of information.
        """
        self.dict_info = {
            constants.IOFields.SUBJECT_ID: self.config.subject_id,
            constants.IOFields.SUBJECT_AGE: self.config.correction.subject_age,
            constants.IOFields.SUBJECT_SEX: self.config.correction.subject_sex,
            constants.IOFields.SUBJECT_HEIGHT: self.config.correction.subject_height,
            constants.IOFields.SCAN_DATE: self.dict_dis[constants.IOFields.SCAN_DATE],
            constants.IOFields.PROCESS_DATE: metrics.process_date(),
            constants.IOFields.SCAN_TYPE: self.config.recon.scan_type,
            constants.IOFields.PIPELINE_VERSION: constants.PipelineVersion.VERSION_NUMBER,
            constants.IOFields.INSTITUTION: self.config.institution,
            constants.IOFields.SYSTEM_VENDOR: self.config.system_vendor,
            constants.IOFields.PHILIPS_VERSION: self.config.recon.philips_software,
            constants.IOFields.SOFTWARE_VERSION: self.dict_dis[
                constants.IOFields.SOFTWARE_VERSION],
            constants.IOFields.GIT_BRANCH: report.get_git_branch(),
            constants.IOFields.REFERENCE_DATA_KEY: self.config.reference_data_key,
            constants.IOFields.BANDWIDTH: self.dict_dis[constants.IOFields.BANDWIDTH],
            constants.IOFields.SAMPLE_TIME: (
                1e6 * self.dict_dis[constants.IOFields.SAMPLE_TIME]
            ),
            constants.IOFields.FA_DIS: self.dict_dis[constants.IOFields.FA_DIS],
            constants.IOFields.FA_GAS: self.dict_dis[constants.IOFields.FA_GAS],
            constants.IOFields.FIELD_STRENGTH: self.dict_dis[
                constants.IOFields.FIELD_STRENGTH
            ],
            constants.IOFields.FLIP_ANGLE_FACTOR: signal_utils.calculate_flipangle_factor(
                self.dict_dis[constants.IOFields.FA_GAS],
                self.dict_dis[constants.IOFields.FA_DIS],
            ),
            constants.IOFields.FOV: self.dict_dis[constants.IOFields.FOV],
            constants.IOFields.XE_DISSOLVED_OFFSET_FREQUENCY: self.dict_dis[
                constants.IOFields.XE_DISSOLVED_OFFSET_FREQUENCY
            ],
            constants.IOFields.GRAD_DELAY_X: self.dict_dis[
                constants.IOFields.GRAD_DELAY_X
            ],
            constants.IOFields.GRAD_DELAY_Y: self.dict_dis[
                constants.IOFields.GRAD_DELAY_Y
            ],
            constants.IOFields.GRAD_DELAY_Z: self.dict_dis[
                constants.IOFields.GRAD_DELAY_Z
            ],
            constants.IOFields.HB_CORRECTION_KEY: self.config.hb_correction_key,
            constants.IOFields.HB: self.config.hb,
            constants.IOFields.RBC_HB_CORRECTION_FACTOR: self.rbc_hb_correction_factor,
            constants.IOFields.MEMBRANE_HB_CORRECTION_FACTOR: self.membrane_hb_correction_factor,
            constants.IOFields.VOL_CORRECTION_KEY: self.config.vol_correction_key,
            constants.IOFields.KERNEL_SHARPNESS: self.config.recon.kernel_sharpness_hr,
            constants.IOFields.N_SKIP_START: self.config.recon.n_skip_start,
            constants.IOFields.N_DIS_REMOVED: len(
                self.dict_dis[constants.IOFields.FIDS_DIS]
            )
            - np.sum(
                recon_utils.get_noisy_projections(
                    data=self.dict_dis[constants.IOFields.FIDS_DIS]
                )
            ),
            constants.IOFields.N_GAS_REMOVED: len(
                self.dict_dis[constants.IOFields.FIDS_GAS]
            )
            - np.sum(
                recon_utils.get_noisy_projections(
                    data=self.dict_dis[constants.IOFields.FIDS_GAS]
                )
            ),
            constants.IOFields.REMOVE_NOISE: self.config.recon.remove_noisy_projections,
            constants.IOFields.SHAPE_FIDS: self.dict_dis[constants.IOFields.FIDS].shape,
            constants.IOFields.SHAPE_IMAGE: self.image_gas_highreso.shape,
            constants.IOFields.T2_CORRECTION_FACTOR_MEMBRANE: signal_utils.calculate_t2star_correction(
                self.dict_dis[constants.IOFields.TE90],
                constants.T2STAR_MEMBRANE_3T,
                self.dict_dis[constants.IOFields.FIELD_STRENGTH],
            ),
            constants.IOFields.T2_CORRECTION_FACTOR_RBC: signal_utils.calculate_t2star_correction(
                self.dict_dis[constants.IOFields.TE90],
                constants.T2STAR_RBC_3T,
                self.dict_dis[constants.IOFields.FIELD_STRENGTH],
            ),
            constants.IOFields.TE90: 1e6 * self.dict_dis[constants.IOFields.TE90],
            constants.IOFields.TR_DIS: 1e3 * self.dict_dis[constants.IOFields.TR_DIS],
        }

        return self.dict_info


    def generate_figures(self):
        """Export image figures."""
        index_start, index_skip = plot.get_plot_indices(self.mask)
        proton_reg = img_utils.normalize(
            np.abs(self.image_proton),
            self.mask,
            method=constants.NormalizationMethods.PERCENTILE,
        )
        plot.plot_montage_grey(
            image=np.abs(self.image_gas_highreso),
            path="tmp/png/montage_vent.png",
            index_start=index_start,
            index_skip=index_skip,
        )
        plot.plot_montage_grey(
            image=np.abs(self.image_membrane),
            path="tmp/png/montage_membrane.png",
            index_start=index_start,
            index_skip=index_skip,
        )
        plot.plot_montage_grey(
            image=np.abs(self.image_rbc),
            path="tmp/png/montage_rbc.png",
            index_start=index_start,
            index_skip=index_skip,
        )
        plot.plot_montage_color(
            image=plot.map_and_overlay_to_rgb(
                self.image_gas_binned, proton_reg, constants.CMAP.VENT_BIN2COLOR
            ),
            path="tmp/png/montage_gas_binned.png",
            index_start=index_start,
            index_skip=index_skip,
        )
        plot.plot_montage_color(
            image=plot.map_and_overlay_to_rgb(
                self.image_rbc2gas_binned, proton_reg, constants.CMAP.RBC_BIN2COLOR
            ),
            path="tmp/png/montage_rbc_binned.png",
            index_start=index_start,
            index_skip=index_skip,
        )
        plot.plot_montage_color(
            image=plot.map_and_overlay_to_rgb(
                self.image_membrane2gas_binned,
                proton_reg,
                constants.CMAP.MEMBRANE_BIN2COLOR,
            ),
            path="tmp/png/montage_membrane_binned.png",
            index_start=index_start,
            index_skip=index_skip,
        )
        plot.plot_montage_color(
            image=plot.overlay_mask_on_image(proton_reg, self.mask.astype("uint8")),
            path="tmp/png/montage_proton_qa.png",
            index_start=index_start,
            index_skip=index_skip,
        )
        plot.plot_montage_color(
            image=plot.overlay_mask_on_image(
                np.abs(self.image_gas_highreso), self.mask.astype("uint8")
            ),
            path="tmp/png/montage_vent_qa.png",
            index_start=index_start,
            index_skip=index_skip,
        )
        plot.plot_montage_color(
            image=plot.overlay_mask_on_image(
                np.abs(self.image_dissolved), self.mask.astype("uint8")
            ),
            path="tmp/png/montage_dissolved_qa.png",
            index_start=index_start,
            index_skip=index_skip,
        )
        plot.plot_histogram(
            data=img_utils.normalize(self.image_gas_cor, self.mask)[
                np.array(self.mask, dtype=bool)
            ].flatten(),
            path="tmp/png/hist_vent.png",
            color=constants.VENTHISTOGRAMFields.COLOR,
            xlim=constants.VENTHISTOGRAMFields.XLIM,
            ylim=constants.VENTHISTOGRAMFields.YLIM,
            num_bins=constants.VENTHISTOGRAMFields.NUMBINS,
            refer_fit=self.reference_data["healthy_histogram_vent_dir"],
            xticks=constants.VENTHISTOGRAMFields.XTICKS,
            yticks=constants.VENTHISTOGRAMFields.YTICKS,
            xticklabels=constants.VENTHISTOGRAMFields.XTICKLABELS,
            yticklabels=constants.VENTHISTOGRAMFields.YTICKLABELS,
            title=constants.VENTHISTOGRAMFields.TITLE,
        )
        plot.plot_histogram(
            data=np.abs(self.image_rbc2gas)[
                np.array(self.mask_vent, dtype=bool)
            ].flatten(),
            path="tmp/png/hist_rbc.png",
            color=constants.RBCHISTOGRAMFields.COLOR,
            xlim=constants.RBCHISTOGRAMFields.XLIM,
            ylim=constants.RBCHISTOGRAMFields.YLIM,
            num_bins=constants.RBCHISTOGRAMFields.NUMBINS,
            refer_fit=self.reference_data["healthy_histogram_rbc_dir"],
            xticks=constants.RBCHISTOGRAMFields.XTICKS,
            yticks=constants.RBCHISTOGRAMFields.YTICKS,
            xticklabels=constants.RBCHISTOGRAMFields.XTICKLABELS,
            yticklabels=constants.RBCHISTOGRAMFields.YTICKLABELS,
            title=constants.RBCHISTOGRAMFields.TITLE,
        )
        plot.plot_histogram(
            data=np.abs(self.image_membrane2gas)[
                np.array(self.mask_vent, dtype=bool)
            ].flatten(),
            path="tmp/png/hist_membrane.png",
            color=constants.MEMBRANEHISTOGRAMFields.COLOR,
            xlim=constants.MEMBRANEHISTOGRAMFields.XLIM,
            ylim=constants.MEMBRANEHISTOGRAMFields.YLIM,
            num_bins=constants.MEMBRANEHISTOGRAMFields.NUMBINS,
            refer_fit=self.reference_data["healthy_histogram_membrane_dir"],
            xticks=constants.MEMBRANEHISTOGRAMFields.XTICKS,
            yticks=constants.MEMBRANEHISTOGRAMFields.YTICKS,
            xticklabels=constants.MEMBRANEHISTOGRAMFields.XTICKLABELS,
            yticklabels=constants.MEMBRANEHISTOGRAMFields.YTICKLABELS,
            title=constants.MEMBRANEHISTOGRAMFields.TITLE,
        )

        plot.plot_histogram_rbc_osc(
            data=self.image_rbc_osc[self.mask_rbc],
            path="tmp/png/hist_rbc_osc.png",
        )
        plot.plot_data_rbc_k0(
            t=np.arange(self.data_rbc_k0.shape[0])
            * self.dict_dis[constants.IOFields.TR],
            data=self.data_rbc_k0,
            path="tmp/png/data_rbc_k0_proc.png",
            high=self.high_indices,
            low=self.low_indices,
        )
        plot.plot_data_rbc_k0(
            t=np.arange(self.data_rbc_k0.shape[0])
            * self.dict_dis[constants.IOFields.TR],
            data=signal_utils.dixon_decomposition(
                self.data_dissolved, self.rbc_m_ratio
            )[0][:, 0],
            path="tmp/png/data_rbc_k0.png",
            high=self.high_indices,
            low=self.low_indices,
        )
        plot.plot_montage_color(
            image=plot.map_and_overlay_to_rgb(
                self.image_rbc_osc_binned, proton_reg, constants.CMAP.RBC_OSC_BIN2COLOR
            ),
            path="tmp/png/montage_rbc_osc_binned.png",
            index_start=index_start,
            index_skip=index_skip,
        )
        if self.config.correction.vc_correction:
            plot.plot_montage_grey(
                image=np.abs(self.relative_vc_map),
                path="tmp/png/relative_vc_map.png",
                index_start=index_start,
                index_skip=index_skip,
            )
            plot.plot_montage_grey(
                image=np.abs(self.correction_map),
                path="tmp/png/correction_map.png",
                index_start=index_start,
                index_skip=index_skip,
            )
        else:
            plot.plot_montage_grey(image=np.zeros_like(self.image_rbc2gas_binned),
                                   path="tmp/png/relative_vc_map.png",
                                   index_start=index_start,
                                   index_skip=index_skip,)
            plot.plot_montage_grey(image=np.zeros_like(self.image_rbc2gas_binned),
                                   path="tmp/png/correction_map.png",
                                   index_start=index_start,
                                   index_skip=index_skip,)

    def generate_pdf(self):
        """Generate HTML and PDF files."""
        # generate individual PDFs
        pdf_list = [os.path.join("tmp", "pdf", pdf)
            for pdf in ["intro.pdf", "clinical_gx.pdf", "grayscale.pdf", "qa.pdf",
                        "osc_imaging_correction.pdf", "clinical_osc.pdf"]
        ] #
        report.intro(self.dict_info, path=pdf_list[0])
        report.clinical_gx(
            {**self.dict_stats, **self.reference_data['reference_stats']},
            path=pdf_list[1],
        )
        report.grayscale(
            {**self.dict_stats, **self.reference_data['reference_stats']},
            path=pdf_list[2],
        )
        report.qa(
            {**self.dict_stats, **self.reference_data['reference_stats']},
            path=pdf_list[3],
        )
        report.osc_imaging_correction(
            {**self.dict_stats, **self.reference_data['reference_stats']},
            path=pdf_list[4],
        )
        report.clinical_osc(
            {**self.dict_stats, **self.reference_data['reference_stats']},
            path=pdf_list[5],
        )
        # combine all PDFs into one report
        path = f"tmp/pdf/{self.config.subject_id}_report.pdf"
        report.combine_pdfs(pdf_list, path)

    def write_stats_to_csv(self):
        """Write statistics to file."""
        ## write to individual subject csv
        io_utils.export_subject_csv(
            {**self.dict_info, **self.dict_stats},
            path=f"tmp/{self.config.subject_id}_stats.csv",
            overwrite=True,
        )
        ## Writing data to a combined csv file
        io_utils.export_subject_csv(
            {**self.dict_info, **self.dict_stats},
            path=f"data/{self.config.data_file_name}.csv")

    def save_subject_to_mat(self):
        """Save the instance variables into a mat file."""
        path = os.path.join("tmp", "mat", self.config.subject_id + ".mat")

        io_utils.export_subject_mat(self, path)

    def save_files(self):
        """Save select images to nifti files and instance variable to mat."""
        proton_reg = img_utils.normalize(np.abs(self.image_proton), self.mask,
                                         method=constants.NormalizationMethods.PERCENTILE,)
        io_utils.export_nii(np.abs(self.image_gas_highreso), "tmp/nii/gas_highreso.nii",
                            self.dict_dis[constants.IOFields.FOV],)
        io_utils.export_nii_4d(plot.map_and_overlay_to_rgb(self.image_gas_binned,
                            proton_reg, constants.CMAP.VENT_BIN2COLOR,),"tmp/nii/gas_rgb.nii",)
        io_utils.export_nii(np.abs(self.image_gas_highsnr), "tmp/nii/gas_highsnr.nii",
                            self.dict_dis[constants.IOFields.FOV],)
        io_utils.export_nii(np.abs(self.image_rbc),"tmp/nii/rbc.nii",
                            self.dict_dis[constants.IOFields.FOV],)
        io_utils.export_nii(self.image_rbc2gas_binned, "tmp/nii/rbc_binned.nii",
                            self.dict_dis[constants.IOFields.FOV],)
        io_utils.export_nii(np.abs(self.image_membrane), "tmp/nii/membrane.nii",
                            self.dict_dis[constants.IOFields.FOV],)
        io_utils.export_nii(np.abs(self.image_membrane2gas), "tmp/nii/membrane2gas.nii",
                            self.dict_dis[constants.IOFields.FOV],)
        io_utils.export_nii(np.abs(self.image_dissolved), "tmp/nii/dissolved.nii",
                            self.dict_dis[constants.IOFields.FOV],)
        io_utils.export_nii(self.mask.astype(float), "tmp/nii/mask_reg.nii",
                            self.dict_dis[constants.IOFields.FOV],)
        io_utils.export_nii(self.mask_vent.astype(float), "tmp/nii/mask_vent.nii",
                            self.dict_dis[constants.IOFields.FOV],)
        io_utils.export_nii(self.image_rbc_osc_binned, "tmp/nii/osc_binned.nii",
                            self.dict_dis[constants.IOFields.FOV],)
        if self.config.recon.recon_proton:
            io_utils.export_nii(np.abs(self.image_proton), "tmp/nii/proton.nii",
                                self.dict_dis[constants.IOFields.FOV],)
            io_utils.export_nii(np.abs(self.image_proton_reg),"tmp/nii/proton_reg.nii",
                                self.dict_dis[constants.IOFields.FOV],)
        io_utils.export_nii_4d(plot.map_and_overlay_to_rgb(self.image_rbc2gas_binned, proton_reg,
                                        constants.CMAP.RBC_BIN2COLOR), "tmp/nii/rbc2gas_rgb.nii",)
        io_utils.export_nii_4d(plot.map_and_overlay_to_rgb(self.image_membrane2gas_binned,
                proton_reg, constants.CMAP.MEMBRANE_BIN2COLOR,),"tmp/nii/membrane2gas_rgb.nii",)
        io_utils.export_nii(self.mask_rbc.astype(float), "tmp/nii/mask_rbc.nii",
                            self.dict_dis[constants.IOFields.FOV],)
        io_utils.export_nii(self.image_rbc_osc * self.mask, "tmp/nii/osc.nii",
                            self.dict_dis[constants.IOFields.FOV],)
        io_utils.export_nii_4d(plot.map_and_overlay_to_rgb(self.image_rbc_osc_binned,
                proton_reg, constants.CMAP.RBC_OSC_BIN2COLOR,),"tmp/nii/osc_rgb.nii",)
        io_utils.export_nii(self.relative_vc_map, "tmp/nii/relative_vc_map.nii",
                            self.dict_dis[constants.IOFields.FOV],)
        io_utils.export_nii(self.correction_map, "tmp/nii/correction_map.nii",
                            self.dict_dis[constants.IOFields.FOV],)

    def save_config_as_json(self):
        """Save subject config .py file as json."""
        io_utils.export_config_to_json(
            self.config,
            "tmp/json/{}_config_gx_osc_imaging.json".format(self.config.subject_id),
        )

    def move_output_files(self):
        """Move output files into dedicated directory."""
        # define files to move
        output_files = (
            "tmp/json/{}_config_gx_osc_imaging.json".format(self.config.subject_id),
            "tmp/mat/{}.mat".format(self.config.subject_id),
            "tmp/pdf/{}_report.pdf".format(self.config.subject_id),
            "tmp/{}_stats.csv".format(self.config.subject_id),
            "tmp/nii/gas_highreso.nii",
            # "tmp/nii/gas_rgb.nii",
            "tmp/nii/mask_reg.nii",
            # "tmp/nii/membrane2gas_rgb.nii",
            "tmp/nii/proton_reg.nii",
            # "tmp/nii/rbc2gas_rgb.nii",
            # "tmp/nii/osc_rgb.nii",
        )
        # move files
        io_utils.move_files(output_files, self.config.data_dir)
