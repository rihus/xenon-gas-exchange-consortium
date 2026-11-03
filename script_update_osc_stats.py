"""Rebuild a combined cohort stats csv from its output folders, at full precision.

RH: written to refresh the gridded oscillation stats after the keyhole outer k-space
scaling fix (7c9b024) and to add new subjects. The combined csv files had been saved
through Excel (rounded values, M/D/YY dates), so every row is rebuilt from the stats
csv of its own run.

For each row of --old_csv the run is found from its subject, correction variant
(hb_correction_key, vol_correction_key, vc_correction) and ventilation method:

    <data_dir>/<prefix>_<variant>/<method>/<subject_id>_stats.csv

(folders as written by script_run_corrections.py). The row is then taken from that
run, column by column. Columns the run csv does not have (written by an older
pipeline version) keep the value of --old_csv. Columns not in --update_cols that
differ from --old_csv are reported, to show whether anything besides the oscillation
stats moved. Rows of --add_subjects are appended in the variant and method order of
script_run_corrections.py. Row order and columns of --old_csv are kept.

Usage:
    python script_update_osc_stats.py --cohort=MATRIX --prefix=gx_osc_gridded \
        --old_csv=data/stats_GX_gridded_MATRIX_15oct2026.csv \
        --new_csv=data/stats_GX_gridded_MATRIX_21oct2026.csv --add_subjects=014
"""

import glob
import importlib
import os

import numpy as np
import pandas as pd
from absl import app, flags

from main import ALL_VENT_METHODS
# RH: also defines the --cohort, --prefix and --variants flags used here
from script_run_corrections import VARIANTS

FLAGS = flags.FLAGS

flags.DEFINE_string("old_csv", "", "combined stats csv to rebuild")
flags.DEFINE_string("new_csv", "", "path of the rebuilt csv")
flags.DEFINE_list("add_subjects", [], "subject config names to append, e.g. 014")
OSC_COLS = [
    "osc_defect", "osc_low", "osc_defectlow", "osc_high", "osc_mean", "osc_stddev",
    "osc_cov", "osc_negative", "key_radius", "rbc_high_snr", "rbc_low_snr",
    "osc_defect_corr", "osc_low_corr", "osc_defectlow_corr", "osc_high_corr",
    "osc_mean_corr", "osc_stddev_corr", "osc_cov_corr", "osc_negative_corr",
]
flags.DEFINE_list("update_cols", OSC_COLS, "columns expected to change (not reported)")
flags.DEFINE_float("rtol", 1e-6, "relative tolerance when comparing with --old_csv")

# columns that differ only in formatting or run time, not compared
SKIP_COMPARE = {"scan_date", "process_date", "git_branch"}


def load_configs() -> dict:
    """Load the cohort subject configs.

    Returns:
        dict of subject_id -> (config name, data_dir).
    """
    out = {}
    for path in sorted(glob.glob(os.path.join("config", FLAGS.cohort, "*.py"))):
        name = os.path.basename(path)[:-3]
        config = importlib.import_module(f"config.{FLAGS.cohort}.{name}").get_config()
        out[str(config.subject_id)] = (name, str(config.data_dir))
    return out


def is_true(value) -> bool:
    return str(value).strip().upper() == "TRUE"


def variant_of(row) -> str:
    """Get the script_run_corrections variant suffix of a stats row."""
    key = (is_true(row["hb_correction_key"]), is_true(row["vol_correction_key"]),
           is_true(row["vc_correction"]))
    for suffix, (hb_key, vol_key, vc) in VARIANTS.items():
        if (is_true(hb_key), is_true(vol_key), vc) == key:
            return suffix
    raise ValueError(f"No variant for {row['subject_id']}: {key}")


def read_run(data_dir: str, suffix: str, method: str, subject_id: str):
    """Read the single-row stats csv of one run, or None if it is missing."""
    path = os.path.join(
        data_dir, f"{FLAGS.prefix}_{suffix}", method, f"{subject_id}_stats.csv"
    )
    if not os.path.isfile(path):
        return None
    return pd.read_csv(path, dtype=str, keep_default_na=False).iloc[-1]


def native(value: str) -> str:
    """Excel TRUE/FALSE back to the pipeline's True/False."""
    return {"TRUE": "True", "FALSE": "False"}.get(value, value)


def same(old: str, new: str) -> bool:
    """Compare two csv values, numerically if both are numbers."""
    if native(old) == new:
        return True
    try:
        return bool(np.isclose(float(old), float(new), rtol=FLAGS.rtol, atol=0))
    except ValueError:
        return False


def main(argv):
    old = pd.read_csv(FLAGS.old_csv, dtype=str, keep_default_na=False)
    configs = load_configs()
    rows, missing, kept_old, diffs = [], [], {}, []

    for _, old_row in old.iterrows():
        subject_id = old_row["subject_id"]
        suffix = variant_of(old_row)
        method = old_row["vent_normalization_method"]
        run = read_run(configs[subject_id][1], suffix, method, subject_id)
        if run is None:
            missing.append(f"{subject_id} {suffix} {method}")
            rows.append(old_row.map(native))
            continue
        row = {}
        for col in old.columns:
            if col not in run.index:
                row[col] = native(old_row[col])
                kept_old[col] = kept_old.get(col, 0) + 1
                continue
            row[col] = run[col]
            if col not in SKIP_COMPARE and col not in FLAGS.update_cols:
                if not same(old_row[col], run[col]):
                    diffs.append((subject_id, suffix, method, col, old_row[col], run[col]))
        rows.append(pd.Series(row))

    n_added = 0
    for name in FLAGS.add_subjects:
        subject_id = next(s for s, (n, _) in configs.items() if n == name)
        if subject_id in set(old["subject_id"]):
            raise ValueError(f"{subject_id} is already in {FLAGS.old_csv}")
        for suffix in FLAGS.variants:
            for method in ALL_VENT_METHODS:
                run = read_run(configs[subject_id][1], suffix, method, subject_id)
                if run is None:
                    missing.append(f"{subject_id} {suffix} {method} (not added)")
                    continue
                rows.append(pd.Series({c: run.get(c, "NA") for c in old.columns}))
                n_added += 1

    pd.DataFrame(rows, columns=old.columns).to_csv(FLAGS.new_csv, index=False)

    print(f"\n{FLAGS.new_csv}: {len(rows)} rows, {n_added} added")
    for m in missing:
        print(f"  run not found, row kept from old csv: {m}")
    if kept_old:
        print("columns not in the run csv, kept from old csv (rows):")
        for col, n in kept_old.items():
            print(f"  {col}: {n}")
    print(f"other columns differing from old csv (rtol {FLAGS.rtol}): {len(diffs)}")
    if diffs:
        d = pd.DataFrame(diffs, columns=["subject", "variant", "method", "col", "old", "new"])
        print(d.groupby("col").size().sort_values(ascending=False).to_string())
        print(d.head(30).to_string(index=False))


if __name__ == "__main__":
    app.run(main)
