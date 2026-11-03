"""Run all cchmc_ILD subjects for both sites.

RH: batch runner for the configs in config/cchmc_ILD/CCHMC and config/cchmc_ILD/Duke.

Each config is run in its own python process (python main.py --config=...), so
a crash in one run cannot stop the batch. Runs are sequential, since the
pipeline writes to the shared tmp folder.

By default each run uses the vent_normalization_method of its config file and
writes directly into the output folder (e.g. gx_osc). --method overrides it on
the command line, the config files are not modified. With --method=all every
method is run into its own subfolder (e.g. gx_osc/glb_99); GLB_FV is then
skipped by main.py for subjects without bag_volume.

--overrides sets other config fields for every run in the same way, e.g. a
different output folder, so a variant can be run next to the existing output.
--cchmc_key_radius sets osc_recon.key_radius (points) for the CCHMC site only
(Duke keeps the value of its config).

Runs that are already complete are skipped unless --rerun is set. Each batch
writes into <batch_log_dir>/<timestamp>/:

    batch_log.txt          one line per run (status, time, error) + summary
    <site>_<subject>.log   full pipeline output of that run
    <site>_<subject>_config.py   config used for that run (only with --overrides)
    combined_stats.csv     stats csv rows of all selected subjects and sites
                           that have output (also from earlier batches)

A run that takes longer than --timeout_min is killed and noted as failed.

Usage:
    python cchmc_ILD_batch_runner.py
    python cchmc_ILD_batch_runner.py --sites=Duke --subjects=HC002,HC051
    python cchmc_ILD_batch_runner.py --method=all
    python cchmc_ILD_batch_runner.py --dry_run
    python cchmc_ILD_batch_runner.py --sites=Duke \
        --overrides=recon.traj_scaling_factor=1,registration_key=mask2gas,output_folder=gx_osc_NoTarjScaling
    python cchmc_ILD_batch_runner.py --cchmc_key_radius=5 \
        --overrides=recon.traj_scaling_factor=0.903,output_folder=gx_osc_TrajScaling0p903_kr8
"""

import glob
import importlib.util
import os
import signal
import subprocess
import sys
import time
from datetime import datetime

import pandas as pd
from absl import app, flags

from utils import constants, io_utils

FLAGS = flags.FLAGS

flags.DEFINE_list("sites", ["CCHMC", "Duke"], "site folders in config/cchmc_ILD")
flags.DEFINE_list("subjects", [], "subject config names, e.g. HC002,HC051 (default: all)")
flags.DEFINE_enum(
    "method",
    None,
    ["glb_99", "glb_fv", "glb_ma", "threshold_ma", "all"],
    "vent normalization method (default: the one in each config file)",
)
flags.DEFINE_list(
    "overrides",
    [],
    "config overrides for every run, as key=value, e.g. recon.traj_scaling_factor=1",
)
# RH: CCHMC readouts (58 pts, 19 us dwell) reach ~2x the k-space radius per point of
# Duke's. RH: upstream now sets key_radius in points (0 = auto, 6.8% k_max), so this
# flag overrides osc_recon.key_radius (points), not the old key_radius_pct
flags.DEFINE_integer(
    "cchmc_key_radius",
    None,
    "osc_recon.key_radius (points per spoke) for CCHMC site runs only (default: from config)",
)
flags.DEFINE_boolean("rerun", False, "rerun subjects that are already complete")
flags.DEFINE_boolean("dry_run", False, "only list what would run")
flags.DEFINE_float("timeout_min", 30, "kill a run after this many minutes (0: no limit)")
flags.DEFINE_string("batch_log_dir", "logs/cchmc_ILD_batch", "folder for the batch logs")

CONFIG_PATH = os.path.join("config", "cchmc_ILD")

# methods of ALL mode, in the order main.py runs them
ALL_VENT_METHODS = [
    constants.NormalizationMethods.GLB_99,
    constants.NormalizationMethods.GLB_FV,
    constants.NormalizationMethods.GLB_MA,
    constants.NormalizationMethods.THRESHOLD_MA,
]


def get_config_paths() -> list:
    """Get the config files to run.

    Returns:
        list of (site, subject, config path), sorted by subject then site.
    """
    runs = []
    for site in FLAGS.sites:
        paths = glob.glob(os.path.join(CONFIG_PATH, site, "*.py"))
        if not paths:
            raise ValueError("No config files found for site: {}".format(site))
        for path in paths:
            subject = os.path.basename(path)[:-3]
            if not FLAGS.subjects or subject in FLAGS.subjects:
                runs.append((site, subject, path))
    found = {subject for _, subject, _ in runs}
    missing = [subject for subject in FLAGS.subjects if subject not in found]
    if missing:
        raise ValueError("No config files found for: {}".format(", ".join(missing)))
    return sorted(runs, key=lambda run: (run[1], run[0]))


def load_config(path: str):
    """Load the config dict of a config file.

    Args:
        path (str): path to the config .py file

    Returns:
        config dict returned by get_config() of the file.
    """
    spec = importlib.util.spec_from_file_location("batch_config", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = module.get_config()
    # RH: apply --overrides here too, so the complete/skip check and
    # combined_stats.csv look in the output folder the run writes to
    for key, value in get_overrides(config, path).items():
        *parents, name = key.split(".")
        node = config
        for parent in parents:
            node = getattr(node, parent)
        setattr(node, name, value)
    return config


def get_override_list(path: str) -> list:
    """Get the overrides of a run as key=value strings.

    Args:
        path (str): path to the config .py file of the run

    Returns:
        --overrides, plus the site-specific overrides for the site of the run.
    """
    overrides = list(FLAGS.overrides)
    site = os.path.basename(os.path.dirname(os.path.abspath(path)))
    if site == "CCHMC" and FLAGS.cchmc_key_radius is not None:
        overrides.append("osc_recon.key_radius={}".format(FLAGS.cchmc_key_radius))
    return overrides


def get_overrides(config, path: str) -> dict:
    """Get the overrides of a run as key: value, typed like the config field.

    Args:
        config: config dict the overrides are for
        path (str): path to the config .py file of the run

    Returns:
        dict of dotted config key: value, e.g. {"recon.traj_scaling_factor": 1.0}.
    """
    overrides = {}
    for override in get_override_list(path):
        # skip empty entries, e.g. from a double comma
        if not override.strip():
            continue
        key, value = override.split("=", 1)
        old = config
        for name in key.split("."):
            old = getattr(old, name)
        if old in (None, constants.NONE):
            # field is unset ("None") in the config: take a number if it is one
            try:
                overrides[key] = float(value)
            except ValueError:
                overrides[key] = value
        else:
            # keep the type of the config field, e.g. "1" -> 1.0
            overrides[key] = type(old)(value)
    return overrides


def write_override_config(path: str, config, override_path: str):
    """Write a config file that is the config at path with --overrides applied.

    RH: main.py cannot take config.recon.* from the command line (recon is not
    a ConfigDict), so the overrides go in through a generated config file. The
    original config file is not modified.

    Args:
        path (str): path to the original config .py file
        config: config dict of the run, to get the types of the fields
        override_path (str): path of the config file to write
    """
    lines = [
        '"""Generated by cchmc_ILD_batch_runner.py: {} with overrides."""'.format(path),
        "import importlib.util",
        "",
        "",
        "def get_config():",
        "    spec = importlib.util.spec_from_file_location(",
        "        'base_config', {!r}".format(os.path.abspath(path)),
        "    )",
        "    module = importlib.util.module_from_spec(spec)",
        "    spec.loader.exec_module(module)",
        "    config = module.get_config()",
    ]
    for key, value in get_overrides(config, path).items():
        lines.append("    config.{} = {!r}".format(key, value))
    lines.append("    return config")
    with open(override_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def get_method(config) -> str:
    """Get the normalization method of a run.

    Args:
        config: config dict of the run

    Returns:
        --method if set, otherwise the method of the config file.
    """
    return FLAGS.method or config.vent_normalization_method


def get_stats_csvs(config) -> dict:
    """Get the stats csv written for each normalization method of a run.

    Args:
        config: config dict of the run

    Returns:
        dict of method: stats csv path, or None if the method has no output yet.
    """
    method = get_method(config)
    if method != constants.NormalizationMethods.ALL:
        # single method: written directly into the output folder
        folder = io_utils.get_output_subfolder(config)
        for path in sorted(glob.glob(os.path.join(folder, "*_stats.csv"))):
            # output of another method in the same folder does not count
            if pd.read_csv(path)["vent_normalization_method"].iloc[0] == method:
                return {method: path}
        return {method: None}
    methods = list(ALL_VENT_METHODS)
    if config.bag_volume in (None, "None", "NA", ""):
        methods.remove(constants.NormalizationMethods.GLB_FV)
    stats_csvs = {}
    for method in methods:
        folder = io_utils.get_output_subfolder(config, method)
        paths = sorted(glob.glob(os.path.join(folder, "*_stats.csv")))
        stats_csvs[method] = paths[0] if paths else None
    return stats_csvs


def get_missing_methods(config) -> list:
    """Get the normalization methods that have not written their stats csv.

    Args:
        config: config dict of the run

    Returns:
        list of methods without a stats csv.
    """
    return [method for method, path in get_stats_csvs(config).items() if path is None]


def get_error_line(run_log: str) -> str:
    """Get the last non-empty line of a run log, usually the exception message.

    Args:
        run_log (str): path to the log file of the run

    Returns:
        the last non-empty line, or "" if the log is empty.
    """
    with open(run_log, errors="replace") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines[-1] if lines else ""


def run_config(path: str, run_log: str) -> tuple:
    """Run one config in its own process.

    Args:
        path (str): path to the config .py file
        run_log (str): path to write the pipeline output of the run to

    Returns:
        (status, note): status is "ok", "skipped", "incomplete" or "failed".
    """
    try:
        config = load_config(path)
    except Exception as error:
        return "failed", "cannot load config: {!r}".format(error)
    if not FLAGS.rerun and not get_missing_methods(config):
        return "skipped", "already complete"
    if get_override_list(path):
        override_path = run_log[:-4] + "_config.py"
        write_override_config(path, config, override_path)
        path = override_path
    command = [sys.executable, "main.py", "--config={}".format(path)]
    if FLAGS.method:
        command.append("--config.vent_normalization_method={}".format(FLAGS.method))
    with open(run_log, "w") as f:
        # own process group, so a timeout also kills the tools started by the run
        process = subprocess.Popen(
            command, stdout=f, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            returncode = process.wait(timeout=FLAGS.timeout_min * 60 or None)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            note = "killed after {:g} min timeout (see {})".format(
                FLAGS.timeout_min, os.path.basename(run_log)
            )
            return "failed", note
        except KeyboardInterrupt:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise
    if returncode != 0:
        note = "exit code {}: {} (see {})".format(
            returncode, get_error_line(run_log), os.path.basename(run_log)
        )
        return "failed", note
    missing = get_missing_methods(config)
    if missing:
        return "incomplete", "no stats csv for: {}".format(", ".join(missing))
    return "ok", ""


def combine_stats(runs: list, path: str) -> int:
    """Combine the stats csv files of all runs into one csv.

    Args:
        runs (list): list of (site, subject, config path)
        path (str): path of the combined csv to write

    Returns:
        number of rows written (one per site, subject and method with output).
    """
    rows = []
    for site, subject, config_path in runs:
        try:
            config = load_config(config_path)
        except Exception:
            continue
        for stats_csv in get_stats_csvs(config).values():
            if stats_csv:
                df = pd.read_csv(stats_csv)
                df.insert(0, "site", site)
                rows.append(df)
    if rows:
        pd.concat(rows, ignore_index=True).to_csv(path, index=False)
    return sum(len(df) for df in rows)


def main(argv):
    """Run all configs and write the batch log."""
    runs = get_config_paths()
    if FLAGS.dry_run:
        for site, subject, path in runs:
            config = load_config(path)
            done = not get_missing_methods(config)
            state = "skip (complete)" if done and not FLAGS.rerun else "run"
            print("{:16s} {:6s} {:6s} {}".format(state, site, subject, path))
        return

    log_dir = os.path.join(FLAGS.batch_log_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(log_dir)
    batch_log = os.path.join(log_dir, "batch_log.txt")
    counts = {}
    start_batch = time.time()

    with open(batch_log, "w") as log:

        def write(line: str):
            # write each line right away, so the log survives a killed batch
            print(line)
            log.write(line + "\n")
            log.flush()

        write("cchmc_ILD batch started {}".format(datetime.now().isoformat(" ", "seconds")))
        write(
            "{} runs, sites: {}, method: {}, rerun: {}".format(
                len(runs), ",".join(FLAGS.sites), FLAGS.method or "from config", FLAGS.rerun
            )
        )
        if FLAGS.overrides:
            write("overrides: {}".format(", ".join(FLAGS.overrides)))
        if FLAGS.cchmc_key_radius is not None:
            write("CCHMC key_radius: {}".format(FLAGS.cchmc_key_radius))
        write("")
        try:
            for i, (site, subject, path) in enumerate(runs, 1):
                print("[{}/{}] running {} {} ...".format(i, len(runs), site, subject))
                start_time = time.time()
                run_log = os.path.join(log_dir, "{}_{}.log".format(site, subject))
                status, note = run_config(path, run_log)
                counts[status] = counts.get(status, 0) + 1
                write(
                    "{}  {:10s} {:6s} {:6s} {:6.0f} s  {}".format(
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        status.upper(),
                        site,
                        subject,
                        time.time() - start_time,
                        note,
                    ).rstrip()
                )
        except KeyboardInterrupt:
            write("")
            write("Batch interrupted by user, the run in progress was not finished.")
        write("")
        summary = ", ".join("{} {}".format(n, status) for status, n in sorted(counts.items()))
        write("Summary: {} of {} runs done ({})".format(sum(counts.values()), len(runs), summary))
        write("Total time: {:.1f} min".format((time.time() - start_batch) / 60))
        n_rows = combine_stats(runs, os.path.join(log_dir, "combined_stats.csv"))
        write("combined_stats.csv: {} rows".format(n_rows))
    print("Batch log: {}".format(batch_log))


if __name__ == "__main__":
    app.run(main)
