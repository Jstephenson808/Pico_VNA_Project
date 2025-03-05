import itertools
import os
import pathlib

from pylint.exceptions import InvalidArgsError

from vna.VNA_utils import (
    get_experiment_plans_folder_path,
    get_experiment_plan_file_path,
)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from itertools import combinations, pairwise
from statistics import mean
from random import random, choice, sample

import numpy as np

from ml_model import *
from VNA_enums import DfFilterOptions
from VNA_utils import (
    mhz_to_hz,
    hz_to_ghz,
    get_frequency_column_headings_list,
    open_pickled_object,
    get_full_df_path,
    open_full_results_df,
)
from matplotlib import pyplot as plt


def extract_report_dictionary_from_test_results(result_dict):
    # need to just get the results, just want the dict keys which contain
    # "report"
    columns = [x for x in result_dict.keys() if "report" in x]
    return extract_gesture_metric_values(result_dict, columns)


def test_data_frame_classifier_frequency_window_with_report(
    data_frame: pd.DataFrame, label: str, frequency_hop: int = mhz_to_hz(100)
) -> pd.DataFrame:
    #
    movement_vector = create_movement_vector_for_single_data_frame(data_frame)
    # as df format is | labels | fq1 | fq2 ......
    # need to get just the fqs which are listed
    freq_list = get_frequency_column_headings_list(data_frame)

    min_frequency, max_frequency = min(freq_list), max(freq_list)
    low_frequency, high_frequency = min_frequency, min_frequency + frequency_hop

    f1_scores = {}

    while high_frequency <= max_frequency:
        print_fq_hop(high_frequency, label, low_frequency)

        #
        data_frame_fq_range_filtered = filter_cols_between_fq_range(
            data_frame, low_frequency, high_frequency
        )
        fq_label = f"{label}_{hz_to_ghz(low_frequency)}_{hz_to_ghz(high_frequency)}"
        result, fname = feature_extract_test_filtered_data_frame(
            data_frame_fq_range_filtered, movement_vector, fname=fq_label
        )
        f1_scores[fq_label] = extract_report_dictionary_from_test_results(result)
        low_frequency += frequency_hop
        high_frequency += frequency_hop
    return pd.DataFrame.from_dict(
        f1_scores, orient="index", columns=[x for x in result.keys() if "report" in x]
    )


def print_fq_hop(high_frequency, label, low_frequency):
    print(f"{label}\n\r{hz_to_ghz(low_frequency)}GHz->{hz_to_ghz(high_frequency)}GHz")


def test_classifier_from_df_dict(
    df_dict: {}, frequency_hop=mhz_to_hz(100), temp_fname="temp.txt"
) -> pd.DataFrame:
    """
    This returns a report and save classifier to pkl path
    """
    full_results_df = None
    with open(get_experiment_plan_file_path(temp_fname), "r") as f:
        parameters = [line.strip().split() for line in f.readlines()]
    for label, data_frame in df_dict.items():
        print(f"testing {label}")
        result_df = test_data_frame_classifier_frequency_window_with_report(
            data_frame, label, frequency_hop=frequency_hop
        )
        full_results_df = pd.concat((full_results_df, result_df))
    return full_results_df


def filter_sparam_combinations(data: pd.DataFrame, *, mag_or_phase) -> {}:
    s_param_dict = {}
    s_param_combs = combinations([param.value for param in SParam2Port], 2)
    for s_param_1, s_param_2 in s_param_combs:
        s_param_dict[f"{s_param_1}_{s_param_2}_{mag_or_phase}"] = data[
            (
                (data[DataFrameCols.S_PARAMETER.value] == s_param_1)
                & (data["mag_or_phase"] == mag_or_phase)
            )
            | (
                (data[DataFrameCols.S_PARAMETER.value] == s_param_2)
                & (data["mag_or_phase"] == mag_or_phase)
            )
        ]
    return s_param_dict


# def create_test_dict(
#     combined_df: pd.DataFrame,
#     sparam_sets: list[list[str]] = None,
#     filter_type: DfFilterOptions = DfFilterOptions.BOTH,
#     temp_txt_file_path=None,
# ) -> dict:
#     if temp_txt_file_path:
#         if pathlib.Path(temp_txt_file_path).exists():
#             return
#         else:
#             raise InvalidArgsError
#     else:
#         return create_test_dict_initial(combined_df, sparam_sets, filter_type)
#
#
# def create_test_dict_with_temp_txt_file(combined_df: pd.DataFrame, temp_txt_file_path):
#     with open(temp_txt_file_path, "r") as f:
#         temp_file_data = [line.strip().split(" ") for line in f.readlines()]
#     split_data = [
#         item[0].rsplit("_", maxsplit=1) + [item[1]] for item in temp_file_data
#     ]
#     # this extracts the sparam sets from the data
#     s_parameter_sets = list(
#         map(list, set(map(tuple, [item[0].split("_") for item in split_data])))
#     )
#     frequency_dict = {}
#     phase_mag_set = set()
#     for item in split_data:
#         s_param_set = item[0]
#         phase_mag = item[1]
#         # set will contain phase mag options
#         phase_mag_set.add(phase_mag)
#         frequency = item[2]
#         if s_param_set not in frequency_dict:
#             frequency_dict[s_param_set] = [frequency]
#         else:
#             frequency_dict[s_param_set].append(frequency)
#     if DfFilterOptions.BOTH.value in phase_mag_set:
#         option = DfFilterOptions.BOTH
#     else:
#         if "magnitude" in phase_mag_set:
#             option = DfFilterOptions.MAGNITUDE
#         else:
#             option = DfFilterOptions.PHASE
#     return {
#         "option": option,
#         "sparam_sets": s_parameter_sets,
#         "freq_dict": frequency_dict,
#     }


def create_test_dict(
    combined_df: pd.DataFrame,
    s_param_to_freq_dict: dict,
    filter_type: DfFilterOptions = DfFilterOptions.BOTH,
    filter_freqs: [int] = None,
) -> dict:
    """
    This function creates the test dict for the classifier, allowing filtering by specific S-parameter sets
    and by magnitude, phase, or both.

    :param combined_df: The combined dataframe containing data.
    :param sparam_sets: A list of lists containing S-parameter strings (e.g., [['S11', 'S12'], ['S21']]).
    :param filter_type: Filter by 'magnitude', 'phase', or 'both'. Defaults to 'both'.
    :return: A dictionary with filtered dataframes.
    """

    # Initialize the dictionary to store filtered dataframes
    filtered_df_dict = {}

    # Check the filter type and set which columns to filter

    all_Sparams_magnitude = combined_df[combined_df["mag_or_phase"] == "magnitude"]
    all_Sparams_phase = combined_df[combined_df["mag_or_phase"] == "phase"]

    # Iterate over each sparameter set provided in sparam_sets
    for set_name_filter, freq_plan in s_param_to_freq_dict.items():
        set_name, filter_type = set_name_filter.split(" ")
        sparam_set = set_name.split("_")

        # Filter for magnitude if specified or 'both'
        if filter_type in ["both", "magnitude"]:
            filtered_all_sparams_magnitude = filter_columns(
                all_Sparams_magnitude, freq_plan
            )
            filtered_df_dict[f"{set_name}_magnitude"] = filtered_all_sparams_magnitude[
                all_Sparams_magnitude[DataFrameCols.S_PARAMETER.value].isin(sparam_set)
            ]

        # Filter for phase if specified or 'both'
        if filter_type in ["both", "phase"]:
            filtered_all_sparams_phase = filter_columns(all_Sparams_phase, freq_plan)
            filtered_df_dict[f"{set_name}_phase"] = filtered_all_sparams_phase[
                all_Sparams_phase[DataFrameCols.S_PARAMETER.value].isin(sparam_set)
            ]

        if filter_type in ["both"]:
            filtered_combined_df = filter_columns(combined_df, freq_plan)
            filtered_df_dict[f"{set_name}_both"] = filtered_combined_df[
                combined_df[DataFrameCols.S_PARAMETER.value].isin(sparam_set)
            ]

    # this will remove any freqs which are not in the set
    if filter_freqs:
        for key, df in enumerate(filtered_df_dict):
            df = filter_columns(df, filter_freqs)
            filtered_df_dict[key] = df

    return filtered_df_dict


def generate_experiment_plan_file(
    sparam_sets,
    fq_hop,
    fq_list,
    experiment_plan_filename="experiment",
    filter_options: [DfFilterOptions] = None,
):
    if filter_options is None:
        filter_options = [DfFilterOptions.MAGNITUDE]

    if not experiment_plan_filename.endswith(".txt"):
        experiment_plan_filename += ".txt"

    experiment_perutations = itertools.product(sparam_sets, filter_options)

    with open(
        os.path.join(get_experiment_plans_folder_path(), f"{experiment_plan_filename}"),
        "w",
    ) as f:
        for sparam_set, filter_option in experiment_perutations:
            min_freq = min(fq_list)
            max_freq = max(fq_list)
            current_freq = min_freq
            while (current_freq + fq_hop) < max_freq:
                f.write(f"{('_').join(sparam_set)} {filter_option} {current_freq}\n")
                current_freq += fq_hop


def test_classifier_for_all_measured_params(
    combined_df: pd.DataFrame,
    s_param_to_freq_dict,
    filter_type: DfFilterOptions,
    fq_hop,
) -> pd.DataFrame:
    """
    return report
    """

    filtered_df_dict = create_test_dict(
        combined_df, s_param_to_freq_dict=s_param_to_freq_dict, filter_type=filter_type
    )

    return test_classifier_from_df_dict(filtered_df_dict, frequency_hop=fq_hop)


# todo refactor this mess
def combine_results_and_test(
    full_df_path, sparam_sets, filter_option=DfFilterOptions.BOTH, csv_label=""
):
    file_name = os.path.basename(full_df_path)
    if file_name.endswith(".pkl"):
        combined_df: pd.DataFrame = open_pickled_object(full_df_path)
    else:
        combined_df = combine_data_frames_from_csv_folder(full_df_path, label=csv_label)

    return test_classifier_for_all_measured_params(
        combined_df, sparam_sets, filter_option
    )


def extract_from_temp_file(file_path):
    with open(file_path, "rt") as f:
        lines = f.readlines()
    s_param_set = set()
    mag_or_phase_set = set()
    freq_set = set()
    s_param_to_freq_dict = {}
    for line in lines:
        s_params, mag_or_phase, frequency = line.strip().split(" ")
        s_param_filter_string = f"{s_params} {mag_or_phase}"
        s_param_set.add(s_params)
        mag_or_phase_set.add(DfFilterOptions(mag_or_phase))
        freq_set.add(int(frequency))
        if s_param_filter_string not in s_param_to_freq_dict:
            s_param_to_freq_dict[s_param_filter_string] = [frequency]
        else:
            s_param_to_freq_dict[s_param_filter_string].append(frequency)
    frequency_hop = int(mean([b - a for a, b in pairwise(sorted(list(freq_set)))]))
    s_param_list = [s_params.split("_") for s_params in list(s_param_set)]
    if (DfFilterOptions.BOTH in mag_or_phase_set) or (len(mag_or_phase_set) > 1):
        mag_or_phase = DfFilterOptions.BOTH
    else:
        mag_or_phase = mag_or_phase_set.pop()
    return s_param_list, frequency_hop, mag_or_phase, s_param_to_freq_dict


# function to run tests on a series of folders which contain results .csvs


if __name__ == "__main__":

    # improve saving of full results so that can happen
    # set up all sparams -> permutations
    #

    s_parameter = "S11"
    mag_or_phase = "magnitude"
    label = "single_LIQUID_DIPOLE_SD1_B"
    full_results_df_fname = "sd1_401_75KHz_full_combined_df_2024_07_24.pkl"

    full_df = open_full_results_df("full_combined_df_2024_08_09.pkl")
    full_df.columns = list(full_df.columns[:5]) + [int(x) for x in full_df.columns[5:]]

    s_param_combinations_list = [["S12", "S13", "S14"], ["S34", "S23", "S42"]]

    freq_hop = mhz_to_hz(100)

    # S21 fist ->

    temp_file_name = full_results_df_fname.split(".pkl")[0] + ".txt"
    temp_file_path = os.path.join(
        get_experiment_plans_folder_path(), f"{temp_file_name}"
    )

    if not os.path.exists(temp_file_path):
        generate_experiment_plan_file(
            sparam_sets=s_param_combinations_list,
            filter_type=DfFilterOptions.BOTH,
            fq_hop=freq_hop,
            fq_list=get_frequency_column_headings_list(full_df),
            experiment_plan_filename=temp_file_name,
        )
    s_param_combinations_list, freq_hop, mag_or_phase, s_param_to_freq_dict = (
        extract_from_temp_file(temp_file_path)
    )

    # #todo need to add svm or dtree label to output dict
    full_results_df = test_classifier_for_all_measured_params(
        full_df, s_param_to_freq_dict, mag_or_phase, mhz_to_hz(100)
    )
    # # combine dfs
    # full_df_fname = os.listdir(os.path.join(get_pickle_path(), "full_dfs"))[0]
    # experiment = "watch_small_antenna_1001_140KHz"
    # full_results_df = combine_results_and_test(os.path.join(get_data_path(), experiment))
    #
    # pickle_object(
    #     full_results_df, path=os.path.join(get_pickle_path(), "classifier_results"), file_name=f"full_results_17_09_patent_exp"
    # )

    # open_pickled_object(
    #     r"C:\Users\js637s.CAMPUS\PycharmProjects\Pico_VNA_Project\pickles\full_results_17_09_patent_exp.pkl"
    # )
