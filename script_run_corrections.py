"""Run each subject of a cohort once per correction variant.

RH: batch runner to compare Hb, gas volume and capillary blood volume corrections.

Each subject is run once per variant, each into its own output folder:

    <prefix>_No_Corr              no correction
    <prefix>_Hb_Corr              Hb correction
    <prefix>_gVol_Corr            gas volume correction
    <prefix>_Hb_gVol_Corr         Hb + gas volume correction
    <prefix>_Hb_gVol_capBl_Corr   Hb + gas volume + capillary blood volume correction
    <prefix>_capBl_Corr           capillary blood volume correction only
    <prefix>_Hb_capBl_Corr        Hb + capillary blood volume correction

The first five run by default; pick others with --variants. A variant is not run
if the subject config lacks its input (hb for Hb, corrected_lung_volume for gVol).

Every run uses ALL ventilation normalization methods, manual vent segmentation
with the subject's mask, gridded (ROBERTSON, default) or compressed sensing
(PLUMMER, --recon_key=plummer) reconstruction, MASK2GAS
registration and oscillation analysis. The subject config files are not modified;
settings are overridden in memory. Runs that are already complete are skipped unless --rerun is set.
Each subject is reconstructed once and every variant runs from a copy; --norecon_once
reconstructs again for each variant.

Usage:
    python script_run_corrections.py --cohort=MATRIX
    python script_run_corrections.py --cohort=MATRIX --subjects=002,003
    python script_run_corrections.py --cohort=MATRIX --recon_key=plummer --prefix=gx_osc_cs
    python script_run_corrections.py --cohort=XeLHC --variants=No_Corr,capBl_Corr
    python script_run_corrections.py --cohort=BI --variants=No_Corr,Hb_Corr,Hb_capBl_Corr
"""

import copy
import glob
import importlib
import logging
import os
import shutil
import time
import traceback

from absl import app, flags

from main import (
    ALL_VENT_METHODS,
    gx_mapping_reconstruction,
    reconstruct_subject,
    run_vent_methods,
)
from utils import constants, io_utils

FLAGS = flags.FLAGS

flags.DEFINE_string("cohort", "MATRIX", "cohort folder name in config folder")
flags.DEFINE_list("subjects", [], "subject config names, e.g. 002,003 (default: all)")
flags.DEFINE_string("prefix", "gx_osc_gridded", "prefix of the output folder names")
flags.DEFINE_string("mask_name", "mask_reg_fixed.nii", "manual mask file in data_dir")
flags.DEFINE_boolean("rerun", False, "rerun variants that are already complete")
# RH: corrections are applied after recon, so one recon can serve all variants
flags.DEFINE_boolean(
    "recon_once", True, "reconstruct each subject once for all variants"
)
# RH: pick gridded or compressed sensing recon from the command line
flags.DEFINE_enum(
    "recon_key",
    constants.ReconKey.ROBERTSON.value,
    [constants.ReconKey.ROBERTSON.value, constants.ReconKey.PLUMMER.value],
    "reconstruction: robertson (gridded) or plummer (compressed sensing)",
)

CONFIG_PATH = "config"
TRACHEA_MASK_NAME = "mask_include_trachea.nii"

HB = constants.HbCorrectionKey
VOL = constants.VolCorrectionKey

# folder suffix: (hb_correction_key, vol_correction_key, vc_correction)
VARIANTS = {
    "No_Corr": (HB.NONE.value, VOL.NONE.value, False),
    "Hb_Corr": (HB.RBC_AND_MEMBRANE.value, VOL.NONE.value, False),
    "gVol_Corr": (HB.NONE.value, VOL.RBC_AND_MEMBRANE.value, False),
    "Hb_gVol_Corr": (HB.RBC_AND_MEMBRANE.value, VOL.RBC_AND_MEMBRANE.value, False),
    "Hb_gVol_capBl_Corr": (
        HB.RBC_AND_MEMBRANE.value,
        VOL.RBC_AND_MEMBRANE.value,
        True,
    ),
    # RH: for cohorts without Hb and/or lung volume
    "capBl_Corr": (HB.NONE.value, VOL.NONE.value, True),
    "Hb_capBl_Corr": (HB.RBC_AND_MEMBRANE.value, VOL.NONE.value, True),
}
DEFAULT_VARIANTS = [
    "No_Corr",
    "Hb_Corr",
    "gVol_Corr",
    "Hb_gVol_Corr",
    "Hb_gVol_capBl_Corr",
]

flags.DEFINE_list("variants", DEFAULT_VARIANTS, "variant folder suffixes to run")


def get_subjects() -> list:
    """Get the subject config names to run.

    Returns:
        list of config names (without .py): --subjects, or all in the cohort folder.
    """
    if FLAGS.subjects:
        return FLAGS.subjects
    paths = glob.glob(os.path.join(CONFIG_PATH, FLAGS.cohort, "*.py"))
    return sorted(os.path.basename(path)[:-3] for path in paths)


def get_variant_config(subject: str, suffix: str):
    """Load a subject config and apply the settings of one correction variant.

    Args:
        subject (str): subject config name, e.g. "002"
        suffix (str): variant folder suffix, a key of VARIANTS

    Returns:
        config dict with the variant settings applied.
    """
    module = "{}.{}.{}".format(CONFIG_PATH, FLAGS.cohort, subject)
    config = importlib.import_module(module).get_config()
    hb_key, vol_key, vc_correction = VARIANTS[suffix]

    # same analysis settings for every run
    config.segmentation_key = constants.SegmentationKey.MANUAL_VENT.value
    config.manual_seg_filepath = os.path.join(str(config.data_dir), FLAGS.mask_name)
    config.vent_normalization_method = constants.NormalizationMethods.ALL
    config.recon.recon_key = FLAGS.recon_key
    # RH: only registers proton to gas here; the manual vent mask is not changed
    config.registration_key = constants.RegistrationKey.MASK2GAS.value
    config.osc_recon.oscillation_analysis = True

    # settings of this correction variant
    config.hb_correction_key = hb_key
    config.vol_correction_key = vol_key
    config.osc_recon.vc_correction = vc_correction
    config.output_folder = "{}_{}".format(FLAGS.prefix, suffix)
    return config


def get_missing_inputs(config) -> list:
    """Get the subject inputs a variant needs but the config does not set.

    Args:
        config: config dict with the variant settings applied

    Returns:
        list of missing config fields, empty if the variant can run.
    """
    missing = []
    if config.hb_correction_key != HB.NONE.value:
        if not (isinstance(config.hb, (int, float)) and config.hb > 0):
            missing.append("hb")
    if config.vol_correction_key != VOL.NONE.value:
        if str(config.corrected_lung_volume).strip().upper() in ("NA", "NONE", ""):
            missing.append("corrected_lung_volume")
    return missing


def is_complete(config) -> bool:
    """Check whether every normalization method has written its stats csv.

    Args:
        config: config dict of the run

    Returns:
        True if the output folder holds a stats csv for each method that runs.
    """
    methods = list(ALL_VENT_METHODS)
    if config.bag_volume in (None, "None", "NA", ""):
        methods.remove(constants.NormalizationMethods.GLB_FV)
    # RH: same folder logic as the pipeline, which may append "_osc"
    folder = io_utils.get_output_subfolder(config)
    return all(glob.glob(os.path.join(folder, m, "*_stats.csv")) for m in methods)


def get_pre_run_status(config):
    """Get the status of a variant that should not be run, or None if it should.

    Args:
        config: config dict with the variant settings applied

    Returns:
        "no_input", "skipped" or "failed", or None if the variant should run.
    """
    missing = get_missing_inputs(config)
    if missing:
        logging.warning("Not run, missing in config: %s", ", ".join(missing))
        return "no_input"
    if is_complete(config) and not FLAGS.rerun:
        return "skipped"
    if not os.path.isfile(config.manual_seg_filepath):
        logging.error("Mask not found: %s", config.manual_seg_filepath)
        return "failed"
    return None


def run_variant(subject: str, suffix: str) -> str:
    """Run one subject with one correction variant.

    Args:
        subject (str): subject config name, e.g. "002"
        suffix (str): variant folder suffix, a key of VARIANTS

    Returns:
        status string: "ok", "skipped", "no_input", "incomplete" or "failed".
    """
    config = get_variant_config(subject, suffix)
    status = get_pre_run_status(config)
    if status:
        return status
    try:
        gx_mapping_reconstruction(config)
    except Exception:
        traceback.print_exc()
        return "failed"
    return "ok" if is_complete(config) else "incomplete"


def run_subject_recon_once(subject: str, suffixes: list) -> dict:
    """Reconstruct a subject once, then run each correction variant from a copy.

    RH: every correction (Hb, volume, capillary blood volume) is applied after
    bias field correction, so the reconstruction is the same for all variants.

    Args:
        subject (str): subject config name, e.g. "002"
        suffixes (list): variant folder suffixes, keys of VARIANTS

    Returns:
        dict of variant suffix to status string, as for run_variant.
    """
    configs = {suffix: get_variant_config(subject, suffix) for suffix in suffixes}
    statuses = {suffix: get_pre_run_status(config) for suffix, config in configs.items()}
    to_run = [suffix for suffix in suffixes if statuses[suffix] is None]
    if not to_run:
        return statuses
    try:
        base = reconstruct_subject(configs[to_run[0]])
    except Exception:
        traceback.print_exc()
        statuses.update({suffix: "failed" for suffix in to_run})
        return statuses
    # segmentation writes the trachea mask into the first variant's folder only
    base_folder = io_utils.get_output_subfolder(base.config)
    for suffix in to_run:
        config = configs[suffix]
        logging.info("Running variant %s", suffix)
        subject_variant = copy.deepcopy(base)
        hb_key, vol_key, vc_correction = VARIANTS[suffix]
        subject_variant.config.hb_correction_key = hb_key
        subject_variant.config.vol_correction_key = vol_key
        subject_variant.config.osc_recon.vc_correction = vc_correction
        subject_variant.config.output_folder = config.output_folder
        folder = io_utils.get_output_subfolder(subject_variant.config)
        trachea_mask = os.path.join(base_folder, TRACHEA_MASK_NAME)
        if folder != base_folder and os.path.isfile(trachea_mask):
            os.makedirs(folder, exist_ok=True)
            shutil.copy(trachea_mask, folder)
        try:
            run_vent_methods(subject_variant, subject_variant.config, recon_osc=True)
            subject_variant.check_git_version()
        except Exception:
            traceback.print_exc()
            statuses[suffix] = "failed"
            continue
        statuses[suffix] = "ok" if is_complete(config) else "incomplete"
    return statuses


def main(argv):
    """Run all subjects with all correction variants and print a summary."""
    unknown = [suffix for suffix in FLAGS.variants if suffix not in VARIANTS]
    if unknown:
        raise ValueError("Unknown variants: {}".format(", ".join(unknown)))
    results = []
    for subject in get_subjects():
        if FLAGS.recon_once:
            logging.info("Processing: %s (one recon, %d variants)", subject, len(FLAGS.variants))
            start_time = time.time()
            statuses = run_subject_recon_once(subject, FLAGS.variants)
            seconds = time.time() - start_time
            for suffix in FLAGS.variants:
                name = "{} {}_{}".format(subject, FLAGS.prefix, suffix)
                results.append((statuses[suffix], name, seconds))
            continue
        for suffix in FLAGS.variants:
            name = "{} {}_{}".format(subject, FLAGS.prefix, suffix)
            logging.info("Processing: %s", name)
            start_time = time.time()
            status = run_variant(subject, suffix)
            results.append((status, name, time.time() - start_time))

    print("\nSummary")
    for status, name, seconds in results:
        print("  {:10s} {:45s} {:5.0f} s".format(status, name, seconds))


if __name__ == "__main__":
    app.run(main)
