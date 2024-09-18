import os.path
from itertools import permutations, combinations

import pandas as pd

from VNA_utils import get_full_dfs_path
from ml_model import *
from VNA_utils import (
    get_data_path,
    ghz_to_hz,
    mhz_to_hz,
    hz_to_ghz,
    get_frequency_column_headings_list,
    get_full_results_df_path, reorder_data_frame_columns
)


def extract_report_dictionary_from_test_results(result_dict):
    # need to just get the results
    columns = [x for x in result_dict.keys() if "report" in x]
    return extract_gesture_metric_values(result_dict, columns)


def test_data_frame_classifier_frequency_window_with_report(
    data_frame: pd.DataFrame, label: str, frequency_hop: int = mhz_to_hz(100)
) -> pd.DataFrame:
    movement_vector = create_movement_vector_for_single_data_frame(data_frame)
    freq_list = get_frequency_column_headings_list(data_frame)
    min_frequency, max_frequency = min(freq_list), max(freq_list)
    low_frequency, high_frequency = min_frequency, min_frequency + frequency_hop
    f1_scores = {}
    while high_frequency <= max_frequency:
        print(f"{label}\n\r{hz_to_ghz(low_frequency)}GHz->{hz_to_ghz(high_frequency)}GHz")

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


def test_classifier_from_df_dict(df_dict: {}) -> pd.DataFrame:
    """
    This returns a report and save classifier to pkl path
    """
    full_results_df = None
    for label, data_frame in df_dict.items():
        print(f"testing {label}")
        result_df = test_data_frame_classifier_frequency_window_with_report(
            data_frame, label, frequency_hop=mhz_to_hz(100)
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



# function to run tests on a series of folders which contain results .csvs

if __name__ == "__main__":
    results = open_pickled_object(os.path.join(get_full_dfs_path(), 'full_combined_df_2024_04_07.pkl'))
    # results = get_full_results_df_from_classifier_pkls(r"H:\James\Classifiers", extract='features')
    # pickle_object(results, path=get_full_results_df_path(), file_name="watch_small_ant")
    #results = pd.concat((open_pickled_object(os.path.join(get_full_results_df_path(), "watch_small_ant.pkl")), open_pickled_object(os.path.join(get_full_results_df_path(), "wfa_full_ex_2.pkl"))), ignore_index=True)
    # pickle_object(
    #     results, path=os.path.join(get_pickle_path(), "classifier_results"), file_name="full_results_single-watch-large-ant.pkl"
    # )
    # results = open_pickled_object(os.path.join(get_pickle_path(), "classifier_results", "full_results_single-watch-large-ant_2.pkl"))
    #todo need to add svm or dtree label to output dict

    # combine dfs
    # full_df_fname = os.listdir(os.path.join(get_pickle_path(), "full_dfs"))[0]
    # experiment = "watch_small_antenna_1001_140KHz"
    # full_results_df = combine_results_and_test(os.path.join(get_data_path(), experiment))
    #
    # pickle_object(
    #     full_results_df, path=os.path.join(get_pickle_path(), "classifier_results"), file_name=f"full_results_{experiment}"
    # )
    # experiment = "watch_small_antenna_2001_140KHz"
    # full_results_df = combine_results_and_test(os.path.join(get_data_path(), experiment))
    #
    # pickle_object(
    #     full_results_df, path=os.path.join(get_pickle_path(), "classifier_results"),
    #     file_name=f"full_results_{experiment}"
    # )
