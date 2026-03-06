"""Make reports."""

import os
import sys
from typing import Any, Dict
import logging

import numpy as np
import pdfkit
import PyPDF2
from git.repo import Repo

from utils import constants
sys.path.append("..")

PDF_OPTIONS = {
    "page-width": 300,
    "page-height": 150,
    "margin-top": 1,
    "margin-right": 0.1,
    "margin-bottom": 0.1,
    "margin-left": 0.1,
    "dpi": 300,
    "encoding": "utf-8",
    "enable-local-file-access": None,
}


def get_git_branch() -> str:
    """Get the current git branch.

    Returns:
        str: current git branch, if not in git repo, return "unknown"
    """
    try:
        return Repo("./").active_branch.name
    except:
        return "unknown"


def format_dict(dict_stats: Dict[str, Any]) -> Dict[str, Any]:
    """Format dictionary for report.

    Rounds values to specified decimal places. If unspecified, rounds to 2 places.
    Args:
        dict_stats (Dict[str, Any]): dictionary of statistics
    Returns:
        Dict[str, Any]: formatted dictionary
    """
    # list of variables to round to 0 decimal places
    list_round_0 = [
        constants.StatsIOFields.VENT_DEFECT_PCT,
        constants.StatsIOFields.VENT_LOW_PCT,
        constants.StatsIOFields.VENT_HIGH_PCT,
        constants.StatsIOFields.RBC_DEFECT_PCT,
        constants.StatsIOFields.RBC_LOW_PCT,
        constants.StatsIOFields.RBC_HIGH_PCT,
        constants.StatsIOFields.MEMBRANE_DEFECT_PCT,
        constants.StatsIOFields.MEMBRANE_LOW_PCT,
        constants.StatsIOFields.MEMBRANE_HIGH_PCT,
    ]
    # list of variables to round to 1 decimal places
    list_round_1 = [
        constants.StatsIOFields.MEMBRANE_SNR,
        constants.StatsIOFields.RBC_SNR,
        constants.StatsIOFields.VENT_SNR,
    ]
    # list of variables to round to 3 decimal places
    list_round_3 = [constants.StatsIOFields.RBC_M_RATIO]
    # list of variables to output to multiply by 100 for readabiltiy
    list_mult_100 = [
        constants.StatsIOFields.RBC_MEAN,
        constants.StatsIOFields.MEMBRANE_MEAN,
        constants.StatsIOFields.RBC_MEDIAN,
        constants.StatsIOFields.MEMBRANE_MEDIAN,
        constants.StatsIOFields.RBC_STDDEV,
        constants.StatsIOFields.MEMBRANE_STDDEV,
    ]

    for key in dict_stats.keys():
        if isinstance(dict_stats[key], float) and key in list_round_0:
            dict_stats[key] = int(np.round(dict_stats[key], 0))
        elif isinstance(dict_stats[key], float) and key in list_round_1:
            dict_stats[key] = np.round(dict_stats[key], 1)
        elif isinstance(dict_stats[key], float) and key in list_round_3:
            dict_stats[key] = np.round(dict_stats[key], 3)
        elif isinstance(dict_stats[key], float) and key in list_mult_100:
            dict_stats[key] = np.round(dict_stats[key] * 100, 2)
        elif isinstance(dict_stats[key], float) and (
            key not in list_round_3
            or key not in list_round_0
            or key not in list_mult_100
        ):
            dict_stats[key] = np.round(dict_stats[key], 2)

    return dict_stats

def clinical_gx(dict_stats: Dict[str, Any], path: str):
    """Make clinical report with colormap images.

    First converts dictionary to html format. Then saves to path.
    Args:
        dict_stats (Dict[str, Any]): dictionary of statistics
        path (str): path to save report
    """
    dict_stats = format_dict(dict_stats)
    current_path = os.path.dirname(__file__)
    path_clinical_gx = os.path.abspath(
        os.path.join(current_path, os.pardir, "assets", "html", "clinical_gx.html")
    )
    path_html = os.path.join("tmp", "html", "clinical_gx.html")
    # write report to html
    with open(path_clinical_gx, "r", encoding= "utf-8") as f:
        file = f.read()
        rendered = file.format(**dict_stats)
    with open(path_html, "w", encoding= "utf-8") as o:
        o.write(rendered)
    # write clinical report to pdf
    pdfkit.from_file(path_html, path, options=PDF_OPTIONS)


def clinical_osc(dict_stats: dict[str, Any], path: str):
    """Make RBC oscillations clinical report.

    First converts dictionary to html format. Then saves to path.
    Args:
        dict_stats (Dict[str, Any]): dictionary of statistics
        path (str): path to save report
    """
    dict_stats = format_dict(dict_stats)
    current_path = os.path.dirname(__file__)
    path_clinical = os.path.abspath(
        os.path.join(
            current_path, os.pardir, "assets", "html", "clinical_osc.html"
        )
    )
    path_html = os.path.join("tmp", "html", "clinical_osc.html")
    # write report to html
    with open(path_clinical, "r", encoding= "utf-8") as f:
        file = f.read()
        rendered = file.format(**dict_stats)
    with open(path_html, "w", encoding= "utf-8") as o:
        o.write(rendered)
    # write clinical report to pdf
    pdfkit.from_file(path_html, path, options=PDF_OPTIONS)


def osc_imaging_correction(dict_stats: dict[str, Any], path: str):
    """Make RBC oscillations corrections report.

    First converts dictionary to html format. Then saves to path.
    Args:
        dict_stats (Dict[str, Any]): dictionary of statistics
        path (str): path to save report
    """
    dict_stats = format_dict(dict_stats)
    current_path = os.path.dirname(__file__)
    path_clinical = os.path.abspath(
        os.path.join(
            current_path, os.pardir, "assets", "html", "osc_imaging_correction.html"
        )
    )
    path_html = os.path.join("tmp", "html", "osc_imaging_correction.html")
    # write report to html
    with open(path_clinical, "r", encoding= "utf-8") as f:
        file = f.read()
        rendered = file.format(**dict_stats)
    with open(path_html, "w", encoding= "utf-8") as o:
        o.write(rendered)
    # write clinical report to pdf
    pdfkit.from_file(path_html, path, options=PDF_OPTIONS)


def grayscale(dict_stats: Dict[str, Any], path: str):
    """Make clinical report with grayscale images.

    First converts dictionary to html format. Then saves to path.
    Args:
        dict_stats (Dict[str, Any]): dictionary of statistics
        path (str): path to save report
    """
    dict_stats = format_dict(dict_stats)
    current_path = os.path.dirname(__file__)
    path_clinical = os.path.abspath(
        os.path.join(current_path, os.pardir, "assets", "html", "grayscale.html")
    )
    path_html = os.path.join("tmp", "html", "grayscale.html")
    # write report to html
    with open(path_clinical, "r", encoding= "utf-8") as f:
        file = f.read()
        rendered = file.format(**dict_stats)
    with open(path_html, "w", encoding= "utf-8") as o:
        o.write(rendered)
    # write clinical report to pdf
    pdfkit.from_file(path_html, path, options=PDF_OPTIONS)


def intro(dict_info: Dict[str, Any], path: str):
    """Make info report.

    First converts dictionary to html format. Then saves to path.
    Args:
        dict_info (Dict[str, Any]): dictionary of statistics
        path (str): path to save report
    """
    dict_info = format_dict(dict_info)
    current_path = os.path.dirname(__file__)
    # logging.info("path in intro html: %s", path)
    path_clinical = os.path.abspath(
        os.path.join(current_path, os.pardir, "assets", "html", "intro.html")
    )
    # logging.info("path to intro html file: %s", path_clinical)
    path_html = os.path.join("tmp", "html", "intro.html")
    # write report to html
    with open(path_clinical, "r", encoding= "utf-8") as f: #
        file = f.read()
        rendered = file.format(**dict_info)
    with open(path_html, "w", encoding= "utf-8") as o: #
        o.write(rendered)
    # write clinical report to pdf
    pdfkit.from_file(path_html, path, options=PDF_OPTIONS)


def qa(dict_stats: Dict[str, Any], path: str):
    """Make quality assurance report.

    First converts dictionary to html format. Then saves to path.
    Args:
        dict_info (Dict[str, Any]): dictionary of statistics
        path (str): path to save report
    """
    dict_stats = format_dict(dict_stats)
    current_path = os.path.dirname(__file__)
    path_clinical = os.path.abspath(
        os.path.join(current_path, os.pardir, "assets", "html", "qa.html")
    )
    path_html = os.path.join("tmp", "html", "qa.html")
    # write report to html
    with open(path_clinical, "r", encoding= "utf-8") as f: #
        file = f.read()
        rendered = file.format(**dict_stats)
    with open(path_html, "w", encoding= "utf-8") as o: #
        o.write(rendered)
    # write clinical report to pdf
    pdfkit.from_file(path_html, path, options=PDF_OPTIONS)


def combine_pdfs(pdf_list: list, path: str):
    """Combine PDFs into one.

    Args:
        pdf_list (list): list of file paths for PDFs to combine
        path (str): output path to save combined PDF to
    """

    # initialize PdfWriter object
    pdf_writer = PyPDF2.PdfWriter()

    # loop over each PDF and add it to combined PDF
    for pdf in pdf_list:
        pdf_reader = PyPDF2.PdfReader(pdf)
        for page in pdf_reader.pages:
            pdf_writer.add_page(page)

    # save combined PDF
    with open(path, "wb") as output_file:
        pdf_writer.write(output_file)


def format_dict_spect(dict_stats: dict[str, Any]) -> dict[str, Any]:
    """Format dictionary for report.

    Rounds values to 2 decimal places.
    Args:
        dict_stats (dict[str, Any]): dictionary of statistics
    Returns:
        dict[str, Any]: formatted dictionary
    """
    list_round_3 = [constants.StatsIOFields.RBC_M_RATIO]
    for key in dict_stats.keys():
        if isinstance(dict_stats[key], float) and key in list_round_3:
            dict_stats[key] = np.round(dict_stats[key], 3)
        elif isinstance(dict_stats[key], float) and key not in list_round_3:
            dict_stats[key] = np.round(dict_stats[key], 2)
    return dict_stats


def format_dict_gx_imaging(dict_stats: Dict[str, Any]) -> Dict[str, Any]:
    """Format dictionary for report.

    Rounds values to specified decimal places. If unspecified, rounds to 2 places.
    Args:
        dict_stats (Dict[str, Any]): dictionary of statistics
    Returns:
        Dict[str, Any]: formatted dictionary
    """
    # list of variables to round to 0 decimal places
    list_round_0 = [
        constants.StatsIOFields.VENT_DEFECT_PCT,
        constants.StatsIOFields.VENT_LOW_PCT,
        constants.StatsIOFields.VENT_HIGH_PCT,
        constants.StatsIOFields.RBC_DEFECT_PCT,
        constants.StatsIOFields.RBC_LOW_PCT,
        constants.StatsIOFields.RBC_HIGH_PCT,
        constants.StatsIOFields.MEMBRANE_DEFECT_PCT,
        constants.StatsIOFields.MEMBRANE_LOW_PCT,
        constants.StatsIOFields.MEMBRANE_HIGH_PCT,
    ]
    # list of variables to round to 3 decimal places
    list_round_3 = [constants.StatsIOFields.RBC_M_RATIO]
    for key in dict_stats.keys():
        if isinstance(dict_stats[key], float) and key in list_round_0:
            dict_stats[key] = int(np.round(dict_stats[key], 0))
        elif isinstance(dict_stats[key], float) and key in list_round_3:
            dict_stats[key] = np.round(dict_stats[key], 3)
        elif isinstance(dict_stats[key], float) and (
            key not in list_round_3 or key not in list_round_0
        ):
            dict_stats[key] = np.round(dict_stats[key], 2)
    return dict_stats


def clinical_spect(dict_stats: dict[str, Any], path: str):
    """Make clinical report with colormap images.

    First converts dictionary to html format. Then saves to path.
    Args:
        dict_stats (dict[str, Any]): dictionary of statistics
        path (str): path to save report
    """
    html_temp_file = "clinical_spectroscopy.html"
    dict_stats = format_dict_spect(dict_stats)
    current_path = os.path.dirname(__file__)
    path_clinical = os.path.abspath(
        os.path.join(current_path, os.pardir, "assets", "html", html_temp_file)
    )
    path_html = os.path.join("tmp", html_temp_file)
    # write report to html
    with open(path_clinical, "r", encoding= "utf-8") as f:
        file = f.read()
        rendered = file.format(**dict_stats)
    with open(path_html, "w", encoding= "utf-8") as o:
        o.write(rendered)
    # write clinical report to pdf
    pdfkit.from_file(path_html, path, options=PDF_OPTIONS)


# def clinical_osc_imaging(stats_dict: dict[str, Any], path: str):
#     """Make clinical report.

#     First converts dictionary to html format. Then saves to path.
#     Args:
#         stats_dict (Dict[str, Any]): dictionary of statistics
#         path (str): path to save report
#     """
#     stats_dict = format_dict(stats_dict)
#     current_path = os.path.dirname(__file__)
#     path_clinical = os.path.abspath(
#         os.path.join(
#             current_path, os.pardir, "assets", "html", "clinical_osc_imaging.html"
#         )
#     )
#     path_html = os.path.join("tmp", "clinical_osc_imaging.html")
#     # write report to html
#     with open(path_clinical, "r", encoding= "utf-8") as f:
#         file = f.read()
#         rendered = file.format(**stats_dict)
#     with open(path_html, "w", encoding= "utf-8") as o:
#         o.write(rendered)
#     # write clinical report to pdf
#     pdfkit.from_file(path_html, path, options=PDF_OPTIONS)


# def osc_imaging_correction(stats_dict: dict[str, Any], path: str):
#     """Make clinical report.

#     First converts dictionary to html format. Then saves to path.
#     Args:
#         stats_dict (Dict[str, Any]): dictionary of statistics
#         path (str): path to save report
#     """
#     stats_dict = format_dict(stats_dict)
#     current_path = os.path.dirname(__file__)
#     path_clinical = os.path.abspath(
#         os.path.join(
#             current_path, os.pardir, "assets", "html", "osc_imaging_correction.html"
#         )
#     )
#     path_html = os.path.join("tmp", "osc_imaging_correction.html")
#     # write report to html
#     with open(path_clinical, "r", encoding= "utf-8") as f:
#         file = f.read()
#         rendered = file.format(**stats_dict)
#     with open(path_html, "w", encoding= "utf-8") as o:
#         o.write(rendered)
#     # write clinical report to pdf
#     pdfkit.from_file(path_html, path, options=PDF_OPTIONS)


# def clinical_gx_imaging(dict_stats: dict[str, Any], path: str):
#     """Make clinical report with colormap images.

#     First converts dictionary to html format. Then saves to path.
#     Args:
#         dict_stats (Dict[str, Any]): dictionary of statistics
#         path (str): path to save report
#     """
#     dict_stats = format_dict(dict_stats)
#     current_path = os.path.dirname(__file__)
#     path_clinical = os.path.abspath(
#         os.path.join(
#             current_path, os.pardir, "assets", "html", "clinical_gx_imaging.html"
#         )
#     )
#     path_html = os.path.join("tmp/", "clinical.html")
#     # write report to html
#     with open(path_clinical, "r", encoding= "utf-8") as f:
#         file = f.read()
#         rendered = file.format(**dict_stats)
#     with open(path_html, "w", encoding= "utf-8") as o:
#         o.write(rendered)
#     # write clinical report to pdf
#     pdfkit.from_file(path_html, path, options=PDF_OPTIONS)


# def grayscale_gx_imaging(dict_stats: Dict[str, Any], path: str):
#     """Make clinical report with grayscale images.

#     First converts dictionary to html format. Then saves to path.
#     Args:
#         dict_stats (Dict[str, Any]): dictionary of statistics
#         path (str): path to save report
#     """
#     dict_stats = format_dict(dict_stats)
#     current_path = os.path.dirname(__file__)
#     path_clinical = os.path.abspath(
#         os.path.join(
#             current_path, os.pardir, "assets", "html", "grayscale_gx_imaging.html"
#         )
#     )
#     path_html = os.path.join("tmp", "grayscale.html")
#     # write report to html
#     with open(path_clinical, "r", encoding= "utf-8") as f:
#         file = f.read()
#         rendered = file.format(**dict_stats)
#     with open(path_html, "w", encoding= "utf-8") as o:
#         o.write(rendered)
#     # write clinical report to pdf
#     pdfkit.from_file(path_html, path, options=PDF_OPTIONS)


# def format_dict(stats_dict: dict[str, Any]) -> dict[str, Any]:
#     """Format dictionary for report.

#     Rounds values to 2 decimal places.
#     Args:
#         stats_dict (Dict[str, Any]): dictionary of statistics
#     Returns:
#         Dict[str, Any]: formatted dictionary
#     """
#     stats_dict = stats_dict.copy()
#     for key in stats_dict.keys():
#         if isinstance(stats_dict[key], (float, np.floating)):
#             stats_dict[key] = round(float(stats_dict[key]), 2)
#     return stats_dict
