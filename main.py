"""Scripts to run gas exchange mapping pipeline."""

import copy

import numpy as np
from absl import app, flags
from ml_collections import config_flags

from config import base_config
from subject_classmap import Subject
from utils import constants

import logging

for n in list(logging.root.manager.loggerDict):
    if n.startswith("fontTools"):
        logging.getLogger(n).setLevel(logging.WARNING)
        logging.getLogger(n).disabled = True
        logging.getLogger(n).propagate = False

FLAGS = flags.FLAGS

_CONFIG = config_flags.DEFINE_config_file("config", None, "config file.")
flags.DEFINE_boolean("force_recon", False, "force reconstruction for the subject")
flags.DEFINE_boolean("force_readin", False, "force read in .mat for the subject")
flags.DEFINE_bool("force_segmentation", False, "run segmentation again.")
flags.DEFINE_string("folder", None, "relative path to subject data folder.")


# RH: methods run, in this order, when vent_normalization_method is ALL
ALL_VENT_METHODS = [
    constants.NormalizationMethods.GLB_99,
    constants.NormalizationMethods.GLB_FV,
    constants.NormalizationMethods.GLB_MA,
    constants.NormalizationMethods.THRESHOLD_MA,
]


def get_vent_methods(config: base_config.Config) -> list:
    """Get the ventilation normalization methods to run.

    Args:
        config (config_dict.ConfigDict): config dict

    Returns:
        list of methods: all four for ALL (GLB_FV skipped if no bag_volume),
            otherwise just the configured one.
    """
    if config.vent_normalization_method != constants.NormalizationMethods.ALL:
        return [config.vent_normalization_method]
    methods = list(ALL_VENT_METHODS)
    # RH: GLB_FV needs bag_volume, skip it instead of failing the whole run
    if config.bag_volume in (None, "None", "NA", ""):
        logging.warning("ALL mode: bag_volume not set, skipping GLB_FV.")
        methods.remove(constants.NormalizationMethods.GLB_FV)
    return methods


# RH: attributes set by Subject.reconstruction_rbc_oscillation, cached across methods
OSC_RECON_FIELDS = (
    "data_rbc_k0",
    "high_indices",
    "low_indices",
    "rbc_m_ratio_high",
    "rbc_m_ratio_low",
    "key_radius",
    "image_dissolved_high",
    "image_dissolved_low",
    "data_dissolved",
)


def reconstruct_rbc_oscillation_cached(subject: Subject, osc_cache: dict):
    """Reconstruct the RBC oscillation images, or reuse them if the inputs match.

    RH: the keyhole recon does not depend on the ventilation normalization method,
    so in ALL mode it only needs to run once per subject and correction setting.

    Args:
        subject (Subject): subject after Hb and volume correction
        osc_cache (dict): cache shared across methods, or None to always reconstruct
    """
    if osc_cache is None:
        subject.reconstruction_rbc_oscillation()
        return
    inputs = (subject.rbc_m_ratio, subject.data_gas, subject.data_dissolved)
    cached = osc_cache.get("inputs")
    if (
        cached is not None
        and cached[0] == inputs[0]
        and np.array_equal(cached[1], inputs[1])
        and np.array_equal(cached[2], inputs[2])
    ):
        logging.info("Reusing RBC oscillation keyhole reconstruction")
        for field in OSC_RECON_FIELDS:
            setattr(subject, field, copy.deepcopy(osc_cache[field]))
        return
    # copy inputs first, since the cs keyhole recon scales data_dissolved in place
    osc_cache["inputs"] = copy.deepcopy(inputs)
    subject.reconstruction_rbc_oscillation()
    for field in OSC_RECON_FIELDS:
        osc_cache[field] = copy.deepcopy(getattr(subject, field))


def run_analysis(subject: Subject, recon_osc: bool, osc_cache: dict = None):
    """Run the steps from gas binning to moving output files.

    Args:
        subject (Subject): subject after bias field correction (or mat read-in)
        recon_osc (bool): whether to reconstruct the RBC oscillation images
        osc_cache (dict): keyhole recon cache shared across methods, or None
    """
    subject.gas_binning()
    subject.dixon_decomposition()
    subject.hb_correction()
    subject.vol_correction()
    subject.dissolved_analysis()
    subject.dissolved_binning()
    if subject.config.osc_recon.oscillation_analysis:
        if recon_osc:
            reconstruct_rbc_oscillation_cached(subject, osc_cache)
        subject.oscillation_analysis()
        subject.oscillation_binning()
    subject.get_statistics()
    subject.get_info()
    subject.save_subject_to_mat()
    subject.write_stats_to_csv()
    subject.generate_figures()
    subject.generate_pdf()
    subject.save_files()
    subject.save_config_as_json()
    subject.move_output_files()


def run_vent_methods(subject: Subject, config: base_config.Config, recon_osc: bool):
    """Run the analysis once, or once per method in ALL mode.

    Args:
        subject (Subject): subject after bias field correction (or mat read-in)
        config (config_dict.ConfigDict): config dict
        recon_osc (bool): whether to reconstruct the RBC oscillation images
    """
    if config.vent_normalization_method != constants.NormalizationMethods.ALL:
        run_analysis(subject, recon_osc)
        return
    osc_cache = {}
    for method in get_vent_methods(config):
        logging.info("ALL mode: running normalization method %s", method)
        # RH: fresh copy per method, since later steps modify subject data in place
        subject_method = copy.deepcopy(subject)
        subject_method.config.vent_normalization_method = method
        subject_method.method_subdir = method
        run_analysis(subject_method, recon_osc, osc_cache)


def gx_mapping_reconstruction(config: base_config.Config):
    """Run the gas exchange mapping pipeline with reconstruction.

    Args:
        config (config_dict.ConfigDict): config dict
    """
    subject = reconstruct_subject(config)
    # RH: per-method steps, looped over all methods in ALL mode
    run_vent_methods(subject, config, recon_osc=True)
    subject.check_git_version()
    logging.info("Complete")


def reconstruct_subject(config: base_config.Config) -> Subject:
    """Read raw data and run the steps up to bias field correction.

    RH: split out so a batch runner can reconstruct once and run several correction
    variants from copies of the subject.

    Args:
        config (config_dict.ConfigDict): config dict

    Returns:
        Subject after bias field correction.
    """
    subject = Subject(config=config)
    subject.clear_temp_file()
    # RH: "all" added
    if config.vent_normalization_method not in ["glb_99", "glb_fv", "glb_ma","threshold_ma", "all"]:
        msg = (
            f"You choose a wrong normalization method: {config.vent_normalization_method}! It has to be: GLB_99, GLB_FV, GLB_MA, THRESHOLD_MA, or ALL"
        )
        raise ValueError(msg)
    try:
        subject.read_twix_files()
    except:
        logging.warning("Cannot read in twix files.")
        try:
            subject.read_mrd_files()
        except:
            raise ValueError("Cannot read in raw data files.")
    subject.calculate_rbc_m_ratio()
    logging.info("Reconstructing images")
    subject.preprocess()
    subject.reconstruction_gas()
    subject.reconstruction_dissolved()
    if config.recon.recon_proton:
        if getattr(subject, "dict_ute", None):
            subject.reconstruction_ute()
        elif config.dicom_proton_dir:
            subject.read_dicom_files()
        else:
            subject.image_proton = np.zeros_like(subject.image_gas_highreso)
    elif config.dicom_proton_dir:
        subject.read_dicom_files()
    else:
        subject.image_proton = np.zeros_like(subject.image_gas_highreso)
    subject.segmentation()
    subject.registration()
    subject.biasfield_correction()
    return subject


def gx_mapping_readin(config: base_config.Config):
    """Run the gas exchange imaging pipeline by reading in .mat file.

    Args:
        config (config_dict.ConfigDict): config dict
    """
    subject = Subject(config=config)
    subject.clear_temp_file()
    # RH: "all" added
    if config.vent_normalization_method not in ["glb_99", "glb_fv", "glb_ma","threshold_ma", "all"]:
        msg = (
            f"You choose a wrong normalization method: {config.vent_normalization_method}! It has to be: GLB_99, GLB_FV, GLB_MA, THRESHOLD_MA, or ALL"
        )
        raise ValueError(msg)
    subject.read_mat_file()
    if FLAGS.force_segmentation:
        subject.segmentation()
    # RH: per-method steps, looped over all methods in ALL mode
    run_vent_methods(subject, config, recon_osc=False)
    subject.check_git_version()
    logging.info("Complete")


def main(argv):
    """Run the gas exchange imaging pipeline.

    Either run the reconstruction or read in the .mat file.
    """
    config = _CONFIG.value
    if FLAGS.folder:
        config.data_dir = FLAGS.folder
    if FLAGS.force_recon:
        logging.info("Gas exchange imaging mapping with reconstruction.")
        gx_mapping_reconstruction(config)
    elif FLAGS.force_readin:
        logging.info("Gas exchange imaging mapping with reconstruction.")
        gx_mapping_readin(config)
    elif config.processes.gx_mapping_recon:
        logging.info("Gas exchange imaging mapping with reconstruction.")
        gx_mapping_reconstruction(config)
    elif config.processes.gx_mapping_readin:
        logging.info("Gas exchange imaging mapping with reconstruction.")
        gx_mapping_readin(config)
    else:
        pass


if __name__ == "__main__":
    app.run(main)
