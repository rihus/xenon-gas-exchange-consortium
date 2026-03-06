"""Scripts to run oscillation mapping pipeline."""

import logging

import numpy as np
from absl import app, flags
from ml_collections import config_flags

from config import base_config
from subject_classmap import Subject

FLAGS = flags.FLAGS

_CONFIG = config_flags.DEFINE_config_file("config", None, "config file.")
flags.DEFINE_boolean("force_recon", False, "force reconstruction for the subject")
flags.DEFINE_boolean("force_readin", False, "force read in .mat for the subject")
flags.DEFINE_bool("force_segmentation", False, "run segmentation again.")


def gx_oscillation_mapping_reconstruction(config: base_config.Config):
    """Run the oscillation mapping pipeline with reconstruction.

    Args:
        config (config_dict.ConfigDict): config dict
    """
    subject = Subject(config=config)
    # try:
    #     subject.read_twix_files()
    # except:
    #     logging.warning("No twix files found.")
    try:
        subject.read_mrd_files()
    except Exception as exc:
        raise ValueError("Cannot read in raw data files.") from exc
    logging.info("Getting RBC:M ratio from static spectroscopy.")
    subject.calculate_rbc_m_ratio()
    logging.info("Reconstructing images")
    subject.preprocess()
    logging.info("Preprocessing done!")
    subject.reconstruction_gas()
    subject.reconstruction_dissolved()
    subject.reconstruction_rbc_oscillation()
    logging.info("RBC Oscillations reconstruction done!")
    if config.recon.recon_proton:
        subject.reconstruction_ute()
        logging.info("UTE reconstruction done!")
    elif config.dicom_proton_dir:
        subject.read_dicom_files()
    else:
        subject.image_proton = np.zeros_like(subject.image_gas_highreso)
    subject.segmentation()
    subject.registration()
    subject.biasfield_correction()
    subject.gas_binning()
    subject.dixon_decomposition()
    subject.hb_correction()
    subject.vol_correction()
    subject.dissolved_analysis()
    subject.dissolved_binning()
    subject.oscillation_analysis()
    subject.oscillation_binning()
    subject.get_statistics()
    subject.get_info()
    # subject.save_subject_to_mat()
    subject.write_stats_to_csv()
    subject.generate_figures()
    subject.generate_pdf()
    subject.save_files()
    subject.save_config_as_json()
    subject.move_output_files()
    logging.info("Complete")


def gx_oscillation_mapping_readin(config: base_config.Config):
    """Run the oscillation imaging pipeline by reading in .mat file.

    Args:
        config (config_dict.ConfigDict): config dict
    """
    subject = Subject(config=config)
    subject.read_mat_file()
    if FLAGS.force_segmentation:
        logging.info("Segmenting Proton Mask")
        subject.segmentation()
    subject.registration()
    subject.biasfield_correction()
    subject.gas_binning()
    subject.dixon_decomposition()
    subject.hb_correction()
    subject.vol_correction()
    subject.dissolved_analysis()
    subject.dissolved_binning()
    subject.oscillation_analysis()
    subject.oscillation_binning()
    subject.get_statistics()
    subject.get_info()
    # subject.save_subject_to_mat()
    subject.write_stats_to_csv()
    subject.generate_figures()
    subject.generate_pdf()
    subject.save_files()
    subject.save_config_as_json()
    subject.move_output_files()
    logging.info("Complete")


def main(argv):
    """Run the Gas exchange and oscillation imaging pipeline.

    Either run the reconstruction or read in the .mat file.
    """
    config = _CONFIG.value
    if FLAGS.force_recon:
        logging.info("Gas exchange and RBC Oscillation imaging mapping with reconstruction.")
        gx_oscillation_mapping_reconstruction(config)
    elif FLAGS.force_readin:
        logging.info("Gas exchange and RBC Oscillation imaging mapping from .mat file")
        gx_oscillation_mapping_readin(config)
    elif config.processes.gx_oscillation_mapping_recon:
        logging.info("Gas exchange and RBC Oscillation imaging mapping with reconstruction.")
        gx_oscillation_mapping_reconstruction(config)
    elif config.processes.gx_oscillation_mapping_readin:
        logging.info("Gas exchange and RBC Oscillation imaging mapping from .mat file")
        gx_oscillation_mapping_readin(config)
    else:
        pass


if __name__ == "__main__":
    app.run(main)
