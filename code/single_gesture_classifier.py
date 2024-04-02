import pandas as pd

from ml_model import *
from VNA_utils import (
    get_data_path,
    ghz_to_hz,
    mhz_to_hz,
    hz_to_ghz,
    get_frequency_column_headings_list,
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
        print(f"{hz_to_ghz(low_frequency)}GHz->{hz_to_ghz(high_frequency)}GHz")

        data_frame_magnitude_filtered = filter_cols_between_fq_range(
            data_frame, low_frequency, high_frequency
        )
        fq_label = f"{label}_{hz_to_ghz(low_frequency)}_{hz_to_ghz(high_frequency)}"
        result, fname = feature_extract_test_filtered_data_frame(
            data_frame_magnitude_filtered, movement_vector, fname=fq_label, n_jobs=0
        )
        f1_scores[label] = extract_report_dictionary_from_test_results(result)
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


def test_classifier_for_all_measured_params(combined_df: pd.DataFrame) -> pd.DataFrame:
    """
    return report
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
    return test_classifier_from_df_dict(filtered_df_dict)


if __name__ == "__main__":
    # results = open_pickled_object(os.path.join(get_pickle_path(), "classifier_results"))
    # stacked = results.stack()
    # combine dfs
    combined_df: pd.DataFrame = open_pickled_object(
        os.path.join(
            get_pickle_path(),
            "full_dfs",
            os.listdir(os.path.join(get_pickle_path(), "full_dfs"))[0],
        )
    )
    # combined_df = combine_data_frames_from_csv_folder(
    #     get_data_path(), label="single-watch-large-ant"
    # )

    full_results_df = test_classifier_for_all_measured_params(combined_df)
    pickle_object(
        full_results_df, os.path.join(get_pickle_path(), "classifier_results")
    )
