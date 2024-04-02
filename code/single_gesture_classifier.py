from ml_model import *
from VNA_utils import get_data_path, ghz_to_hz, mhz_to_hz, hz_to_ghz, get_frequency_column_headings_list

if __name__ == "__main__":
    # combine dfs
    combined_df: pd.DataFrame = open_pickled_object(
         os.path.join(get_pickle_path(), "full_dfs", os.listdir(os.path.join(get_pickle_path(), "full_dfs"))[0]))
    #combined_df = combine_data_frames_from_csv_folder(
    #     get_data_path(), label="single-watch-large-ant"
    # )

    all_Sparams_magnitude = combined_df[
        (combined_df["mag_or_phase"] == "magnitude")
    ]
    all_Sparams_phase = combined_df[
        (combined_df["mag_or_phase"] == "phase")
    ]
    test_dfs = {param.value:all_Sparams_magnitude[all_Sparams_magnitude[DataFrameCols.S_PARAMETER.value] == param.value] for param in SParam}
    movement_vector = create_movement_vector_for_single_data_frame(combined_df)
    freq_list = get_frequency_column_headings_list(all_Sparams_magnitude)
    min_frequency, max_frequency = ghz_to_hz(min(freq_list)), ghz_to_hz(max(freq_list))
    low_frequency, high_frequency = min_frequency, min_frequency + mhz_to_hz(100)
    while high_frequency <= max_frequency:
        print(f"{hz_to_ghz(low_frequency)}GHz->{hz_to_ghz(high_frequency)}GHz")

        all_Sparams_magnitude_filtered = filter_cols_between_fq_range(
            all_Sparams_magnitude, low_frequency, high_frequency
        )
        label = f"all_Sparams_magnitude_{hz_to_ghz(low_frequency)}_{hz_to_ghz(high_frequency)}"
        result = feature_extract_test_filtered_data_frame(
            all_Sparams_magnitude_filtered,
            movement_vector,
            fname=label,
        )
        columns = [x for x in result.keys() if "report" in x]
        f1_scores[label] = extract_gesture_metric_values(
            result, columns
        )
        low_frequency += mhz_to_hz(100)
        high_frequency += mhz_to_hz(100)
    full_results_df = pd.DataFrame.from_dict(f1_scores, orient="index", columns=columns)