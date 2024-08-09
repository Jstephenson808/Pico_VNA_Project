import random
import re
from itertools import permutations, combinations

import numpy as np
import pandas as pd

from ml_model import *
from VNA_utils import (
    get_data_path,
    ghz_to_hz,
    mhz_to_hz,
    hz_to_ghz,
    get_frequency_column_headings_list,
    get_full_results_df_path,
    reorder_data_frame_columns,
    get_frequency_column_headings_list
    )
from matplotlib import pyplot as plt


def extract_report_dictionary_from_test_results(result_dict):
    # need to just get the results, just want the dict keys which contain
    # "report"
    columns = [x for x in result_dict.keys() if "report" in x]
    return extract_gesture_metric_values(result_dict, columns)


def test_data_frame_classifier_frequency_window_with_report(
    data_frame: pd.DataFrame, label: str, frequency_hop: int = mhz_to_hz(200)
) -> pd.DataFrame:
    #
    movement_vector = create_movement_vector_for_single_data_frame(data_frame)
    # as df format is | labels | fq1 | fq2 ......
    # need to get just the fqs which are listed
    freq_list = get_frequency_column_headings_list(data_frame)


    min_frequency, max_frequency = min(freq_list), max(freq_list)

    min_frequency= mhz_to_hz(1000)
    max_frequency= mhz_to_hz(1200)

    low_frequency, high_frequency = min_frequency, min_frequency + frequency_hop

    f1_scores = {}

    while high_frequency <= max_frequency:
        print_fq_hop(high_frequency, label, low_frequency)

        #
        data_frame_magnitude_filtered = filter_cols_between_fq_range(
            data_frame, low_frequency, high_frequency
        )
        fq_label = f"{label}_{hz_to_ghz(low_frequency)}_{hz_to_ghz(high_frequency)}"
        result, fname = feature_extract_test_filtered_data_frame(
            data_frame_magnitude_filtered, movement_vector, fname=fq_label
        )
        f1_scores[fq_label] = extract_report_dictionary_from_test_results(result)
        low_frequency += frequency_hop
        high_frequency += frequency_hop
    return pd.DataFrame.from_dict(
        f1_scores, orient="index", columns=[x for x in result.keys() if "report" in x]
    )


def print_fq_hop(high_frequency, label, low_frequency):
    print(f"{label}\n\r{hz_to_ghz(low_frequency)}GHz->{hz_to_ghz(high_frequency)}GHz")


def test_classifier_from_df_dict(df_dict: {}) -> pd.DataFrame:
    """
    This returns a report and save classifier to pkl path
    """
    full_results_df = None
    for label, data_frame in df_dict.items():
        print(f"testing {label}")
        result_df = test_data_frame_classifier_frequency_window_with_report(
            data_frame, label, frequency_hop=mhz_to_hz(200)
        )
        full_results_df = pd.concat((full_results_df, result_df))
    return full_results_df

def filter_sparam_combinations(data: pd.DataFrame, *, mag_or_phase) -> {}:
    s_param_dict = {}
    s_param_combs = combinations([param.value for param in SParam], 2)
    for s_param_1, s_param_2 in s_param_combs:
        s_param_dict[f"{s_param_1}_{s_param_2}_{mag_or_phase}"] = data[((data[DataFrameCols.S_PARAMETER.value]==s_param_1)&(data["mag_or_phase"]==mag_or_phase))|((data[DataFrameCols.S_PARAMETER.value]==s_param_2)&(data["mag_or_phase"]==mag_or_phase))]
    return s_param_dict

def create_test_dict(combined_df: pd.DataFrame, *, test_permuations='ALL')->{}:
    """
    this function creates the test dict for the classifier
    :param combined_df:
    :param test_permuations:
    :return:
    """
    all_Sparams_magnitude = combined_df[(combined_df["mag_or_phase"] == "magnitude")]
    all_Sparams_phase = combined_df[(combined_df["mag_or_phase"] == "phase")]
    filtered_df_dict = {
        f"{param.value}_magnitude": all_Sparams_magnitude[
            all_Sparams_magnitude[DataFrameCols.S_PARAMETER.value] == param.value
            ]
        for param in SParam
    }
    filtered_df_dict["all_Sparams_magnitude"] = all_Sparams_magnitude
    filtered_df_dict["all_Sparams_phase"] = all_Sparams_phase
    filtered_df_dict.update(
        {
            f"{param.value}_phase": all_Sparams_phase[
                all_Sparams_phase[DataFrameCols.S_PARAMETER.value] == param.value
                ]
            for param in SParam
        }
    )
    filtered_df_dict.update(
        filter_sparam_combinations(combined_df, mag_or_phase='magnitude')
    )
    filtered_df_dict.update(
        filter_sparam_combinations(combined_df, mag_or_phase='phase')
    )
    return filtered_df_dict

def test_classifier_for_all_measured_params(combined_df: pd.DataFrame) -> pd.DataFrame:
    """
    return report
    """
    filtered_df_dict = create_test_dict(combined_df, test_permuations='ALL')
    return test_classifier_from_df_dict(filtered_df_dict)

#todo refactor this mess
def combine_results_and_test(full_df_path, csv_label=""):
    file_name = os.path.basename(full_df_path)
    if file_name.endswith(".pkl"):
        combined_df: pd.DataFrame = open_pickled_object(
            full_df_path
        )
    else:
        combined_df = combine_data_frames_from_csv_folder(
            full_df_path, label=csv_label
        )

    return test_classifier_for_all_measured_params(combined_df)

def plot_comparison_table(full_df,*, s_parameter=None, mag_or_phase=None, target_frequency=None):
    if target_frequency is None:
        raise AttributeError("No target frequency")

    if s_parameter is None or mag_or_phase is None:
        raise AttributeError(
            f"Must include all params s_param={s_parameter}, mag_or_phase={mag_or_phase}")

    filtered_df = full_df.query(
        f's_parameter == "{s_parameter}" and mag_or_phase == "{mag_or_phase}"')

    grouped_by_label = filtered_df.groupby('id')
    dfs_to_plot = []
    for label_group in grouped_by_label:

        grouped_by_ids = label_group.groupby('id')
        random_test = random.choice(list(grouped_by_ids.groups.keys()))
        dfs_to_plot.append(filter_fq_cols(filtered_df.query(f'id == "{random_test}"'), target_frequency))

    fig, axs = plt.subplots(nrows=len(dfs_to_plot), ncols=1, sharex=True)

    for ax, df in zip(axs, dfs_to_plot):
        closest_fq = get_closest_freq_column(df, target_frequency)
        ax.plot(df[DataFrameCols.TIME.value], df[closest_fq])
        ax.set_ylabel(f"|{s_parameter}|")
        ax.set_xlabel("Time (s)")
        plt.title(f"|{s_parameter}| Over Time at {hz_to_ghz(closest_fq)} GHz")
        plt.show()



# function to run tests on a series of folders which contain results .csvs

def plot_fq_time_series(full_df: pd.DataFrame,*, s_parameter=None, mag_or_phase=None, label=None, n_random_ids=1, target_frequency=None):
    if target_frequency is None:
        raise AttributeError("No target frequency")

    if s_parameter is None or mag_or_phase is None or label is None:
        raise AttributeError(f"Must include all params s_param={s_parameter}, mag_or_phase={mag_or_phase}, label={label}")

    filtered_df = full_df.query(
        f's_parameter == "{s_parameter}" and mag_or_phase == "{mag_or_phase}" and label == "{label}"')
    grouped_by_id = filtered_df.groupby('id')
    filtered_dfs = []
    random_ids = random.sample(list(grouped_by_id.groups.keys()), n_random_ids)
    random_experiment_list = [filtered_df.query(f'id == "{id}"') for id in random_ids]
    for random_experiment_df in random_experiment_list:
        filtered_dfs.append(filter_fq_cols(random_experiment_df, target_frequency))

    fig, ax = plt.subplots()

    for filtered_df in filtered_dfs:
        closest_fq = get_closest_freq_column(filtered_df, target_frequency)
        ax.plot(
            filtered_df[DataFrameCols.TIME.value],
            filtered_df[closest_fq],
        )
    ax.set_ylabel(f"|{s_parameter}|")
    ax.set_xlabel("Time (s)")
    plt.title(f"|{s_parameter}| Over Time at {hz_to_ghz(closest_fq)} GHz")
    plt.show()


def filter_fq_cols(df, target_frequency):
    closest_fq_col = get_closest_freq_column(df, target_frequency)
    # this filters all teh columns
    # fq_cols = [int(col) for col in [col for col in test.columns.astype(str) if re.search(r'\d', col)]]
    str_cols = [col for col in df.columns if type(col) is str]
    str_cols.append(closest_fq_col)
    return df[str_cols]


def get_closest_freq_column(data_frame, target_frequency):
    fq_series = get_frequency_column_headings_list(data_frame)
    closest_fq_col = fq_series[(np.abs(np.asarray(fq_series) - target_frequency)).argmin()]
    return closest_fq_col


if __name__ == "__main__":
    experiment = "single_Test_dipole2"
    full_results_df = combine_results_and_test(os.path.join(get_data_path(), experiment))
