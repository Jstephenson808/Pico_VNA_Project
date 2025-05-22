import os

import pandas as pd
import random

from sklearn.metrics import confusion_matrix

from vna.VNA_defaults import CONFIRM_TEMP_FILE
from vna.VNA_utils import (
    open_pickled_object,
    convert_magnitude_rows_to_db,
    get_results_path,
    get_full_results_df_path,
    get_pickle_path,
    mhz_to_hz,
    hz_to_mhz,
    get_experiment_plans_folder_path,
    get_frequency_column_headings_list,
    filter_results_df_between_times,
    extract_random_single_gesture_for_each_experiment_to_df,
    coalesce_duplicate_columns,
)
from vna.VNA_enums import (
    MagnitudeOrPhase,
    DfFilterOptions,
    DataFrameCols,
    SParam,
    DfAxis,
)
from vna.VNA_utils import (
    open_pickled_object_in_pickle_folder,
    pickle_object,
)
from vna.graphs import (
    plot_multiple_gestures_on_time_series,
    display_confusion_matrix_for_top_n_values,
    bar_graph_accuracy_comparison,
    get_s_param_data,
    plot_3d_plots,
)
from vna.ml_model import (
    get_full_results_df_from_classifier_pkls,
)
from vna.single_gesture_classifier import (
    generate_experiment_plan_file,
    extract_from_temp_file,
    test_classifier_for_all_measured_params,
)
from vna.touchstoneConverter import TouchstoneConverter

target_sparam = "gloveExperiment_S21"


def convert_touchstones():
    path = r"C:\Users\2573758S\OneDrive - University of Glasgow\PhD\Experiments\Glove Gesture Experiment\Touchstones\Live Capture Touchstones"

    converter = TouchstoneConverter(touchstone_folder_path=path)
    converter.extract_all_touchstone_data_to_dataframe()

    pickle_object(
        converter.output_data_frame, file_name="glove_experiment_results_correct"
    )


def plot_confusion_matrix(target_s_param):
    full_data_frame: pd.DataFrame = open_pickled_object_in_pickle_folder(
        "glove_experiment_singles_only.pkl"
    )
    glove_experiment_classification_results = open_pickled_object(
        r"C:\Users\js637s.CAMPUS\PycharmProjects\Pico_VNA_Project\pickles\full_classification_results\glove_experiment.pkl"
    )
    glove_confusion_matrix = open_pickled_object(
        r"C:\Users\js637s.CAMPUS\PycharmProjects\Pico_VNA_Project\pickles\confusion_matrix\glove_experiment_confusion_matrix.pkl"
    )

    s_param_filtered = glove_experiment_classification_results[
        glove_experiment_classification_results["s_param"] == target_sparam
    ]

    display_confusion_matrix_for_top_n_values(
        full_data_frame,
        s_param_filtered,
        glove_confusion_matrix,
        n=5,
        convert_to_percent_of_true_labels=True,
    )


def process_results(classifier_folder_path):
    results = get_full_results_df_from_classifier_pkls(classifier_folder_path)
    pickle_object(
        results,
        folder_path=get_full_results_df_path(),
        file_name=os.path.basename(classifier_folder_path),
    )
    results.to_csv(os.path.join(get_results_path(), classifier_folder_path.basename()))
    data_frame_to_plot = convert_magnitude_rows_to_db(results)
    plot_multiple_gestures_on_time_series()
    # plot_3d_plots(
    #     results_df,
    # )


def combine_classifier_results_dfs(classification_results_folder):
    classifier_result_fnames = os.listdir(classification_results_folder)
    combined_results_list = []
    for classifier_result_fname in classifier_result_fnames:
        results_df = open_pickled_object(
            os.path.join(classification_results_folder, classifier_result_fname)
        )
        combined_results_list.append(results_df)

    combined_results_df = pd.concat(combined_results_list)
    return combined_results_df


def run_classification_from_results():
    try:
        results_df = open_pickled_object_in_pickle_folder(EXPERIMENT_NAME)
    except FileNotFoundError:
        # extract repeats to df
        converter = TouchstoneConverter(touchstone_folder_path=TOUCHSTONE_FOLDER_PATH)
        converter.extract_all_touchstone_data_to_dataframe()

        results_df = converter.output_data_frame
        pickle_object(
            results_df,
            folder_path=get_pickle_path(),
            file_name=EXPERIMENT_NAME,
        )

    reps_25 = results_df[results_df[DataFrameCols.ID.value].str.contains("25_reps")]
    reps_50 = results_df[results_df[DataFrameCols.ID.value].str.contains("50_reps")]
    results_df = pd.concat(
        [reps_25.dropna(axis="columns"), reps_50.dropna(axis="columns")]
    ).reset_index(drop=True)
    results_df[DataFrameCols.LABEL.value] = (
        results_df[DataFrameCols.LABEL.value]
        .str.replace("_25_reps_", "_")
        .str.replace("_50_reps_", "_")
    )
    pickle_object(
        results_df,
        folder_path=get_pickle_path(),
        file_name="glove_gesture_experiment_2_201pts_75reps_150M_400M_11ges",
    )

    label = "gloveExperiment2"

    s_param_combinations_list = [
        ["S11"],
        ["S21"],
        ["S21", "S11"],
        ["S21", "S31", "S41"],
        ["S21", "S31"],
        ["S21", "S41"],
    ]
    phase_mag = [DfFilterOptions.MAGNITUDE, DfFilterOptions.PHASE, DfFilterOptions.BOTH]
    fq_hops = [mhz_to_hz(i) for i in range(10, 21, 4)]

    for fq_hop in fq_hops:
        temp_file_name = EXPERIMENT_NAME + f"_{hz_to_mhz(fq_hop)}MHz" + ".txt"
        experiment_plan_file_path = os.path.join(
            get_experiment_plans_folder_path(), f"{temp_file_name}"
        )

        if not os.path.exists(experiment_plan_file_path):
            generate_experiment_plan_file(
                sparam_sets=s_param_combinations_list,
                fq_hop=fq_hop,
                fq_list=get_frequency_column_headings_list(results_df),
                experiment_plan_filename=temp_file_name,
                filter_options=phase_mag,
            )
        if CONFIRM_TEMP_FILE:
            choice = input(
                f"Experiment will continue with the experiment plan located at: "
                f"\n{experiment_plan_file_path} "
                f"\ntype N to cancel this and generate a new one,"
                f"\nor press any other key to continue...................."
            )
            if choice == "N":
                generate_experiment_plan_file(
                    sparam_sets=s_param_combinations_list,
                    fq_hop=fq_hop,
                    fq_list=get_frequency_column_headings_list(results_df),
                    experiment_plan_filename=temp_file_name,
                    filter_options=phase_mag,
                )
        s_param_combinations_list, freq_hop, mag_or_phase, s_param_to_freq_dict = (
            extract_from_temp_file(experiment_plan_file_path)
        )

        full_results_df = test_classifier_for_all_measured_params(
            results_df,
            s_param_to_freq_dict,
            fq_hop=freq_hop,
            experiment_plan_path=temp_file_name,
        )
        # combine dfs
        # full_df_fname = os.listdir(os.path.join(get_pickle_path(), "full_dfs"))[0]
        # experiment = "watch_small_antenna_1001_140KHz"
        # full_results_df = combine_results_and_test(os.path.join(get_data_path(), experiment))

        pickle_object(
            full_results_df,
            folder_path=os.path.join(get_pickle_path(), "classifier_results"),
            file_name=f"{label}_{fq_hop}MHz_results.pkl",
        )


if __name__ == "__main__":

    CLASSIFIER_RESULTS_PATH = r"C:\Users\js637s.CAMPUS\PycharmProjects\Pico_VNA_Project\pickles\classifier_results"
    PKL_RESULTS_FNAME = "classification_results_glove_experiment"
    EXPERIMENT_NAME = "glove_gesture_experiment_2_201pts_75reps_150M_400M_11ges"
    TOUCHSTONE_FOLDER_PATH = r"C:\Users\2573758S\OneDrive - University of Glasgow\PhD\Experiments\Glove Gesture Experiment\Touchstones\Experiment 2"
    RESULTS_WITHOUT_REPEATS_PKL_FNAME = "glove_experiment_singles_only.pkl"

    MAGNITUDE_REPEAT_FNAME = r"glove_experiment_results_correct.pkl"
    low_freq = mhz_to_hz(118)
    high_freq = mhz_to_hz(350)
    percent_of_result_to_plot = 20
    start_time = 0
    end_time = 7

    confusion_matrix_target_parameter = "gloveExperiment_S21"

    label = "gloveExperiment2"

    results_df_from_file = combine_classifier_results_dfs(CLASSIFIER_RESULTS_PATH)
    results_from_pkls = open_pickled_object_in_pickle_folder(PKL_RESULTS_FNAME)

    # run_classification_from_results()

    # plot_confusion_matrix(target_s_param=confusion_matrix_target_parameter)

    # 3d plots
    results_df = open_pickled_object_in_pickle_folder(MAGNITUDE_REPEAT_FNAME)

    time_series_to_3d_plot = filter_results_df_between_times(
        results_df, start_time, end_time
    )
    cols_to_drop = list(
        filter(lambda x: (low_freq > x) | (x > high_freq), results_df.columns[5:])
    )
    time_series_to_3d_plot = coalesce_duplicate_columns(time_series_to_3d_plot)
    time_series_to_3d_plot.drop(cols_to_drop, axis=DfAxis.ROW, inplace=True)
    #
    for label, df in time_series_to_3d_plot.groupby("id"):
        for sparam in SParam:
            plot_3d_plots(
                df,
                sparam,
                MagnitudeOrPhase.Magnitude,
                save_to_file=True,
                experiment_label=label,
            )

    results_without_repeat = open_pickled_object_in_pickle_folder(
        RESULTS_WITHOUT_REPEATS_PKL_FNAME
    )

    # results_without_repeat = convert_magnitude_cols_to_db(results_without_repeat)
    experiments = results_without_repeat[DataFrameCols.ID.value].unique()
    s_param = SParam.S11
    mag_or_phase = MagnitudeOrPhase.Magnitude

    single_gesture_for_each_experiment_df = (
        extract_random_single_gesture_for_each_experiment_to_df(
            capture_df=results_without_repeat,
            target_s_param=s_param,
            mag_or_phase=mag_or_phase,
        )
    )

    random_single_gesture_df = single_gesture_for_each_experiment_df[
        single_gesture_for_each_experiment_df[DataFrameCols.ID.value]
        == random.choice(
            single_gesture_for_each_experiment_df[DataFrameCols.ID.value].unique()
        )
    ]
    random_single_gesture_df = random_single_gesture_df.reset_index(drop=True)
    chosen_gesture = random_single_gesture_df["label"][0].split("_")[-1]
    stop_index = 100

    group = random_single_gesture_df.iloc[:, 4:stop_index].groupby("time")

    # plot time series plots

    experiment_name = "2412181557_liquid_metal_glove_6ges_25rps"
    gesture_repeated = full_data_frame.query(f"id == '{experiment_name}'")

    for target_s_param in target_s_params:
        plot_multiple_gestures_on_time_series(
            data_frame=gesture_repeated,
            experiment_label=experiment_name,
            gestures=gestures,
            target_s_param=target_s_param,
            mag_or_phase=MagnitudeOrPhase.Phase,
            target_frequency=target_frequency,
            n_random_ids=5,
        )

    # for plot_label in plot_labels:
    #     for target_s_param in target_s_params:
    #         plot_fq_time_series(
    #             gesture_repeated,
    #             s_parameter=target_s_param,
    #             mag_or_phase=MagnitudeOrPhase.Magnitude,
    #             label=plot_label,
    #             n_random_ids=1,
    #             target_frequency=mhz_to_hz(200),
    #         )

    s11 = full_data_frame.query(
        "id == 'liquid_metal_glove_6ges_same_gesture_10time_2412181543' & s_parameter == 'S11' & label == 'liquid_metal_glove_6ges_same_gesture_10time_1' & mag_or_phase == 'magnitude'"
    )
    full_data_frame.query("")
