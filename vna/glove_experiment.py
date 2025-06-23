import os
from pathlib import Path

import pandas as pd
import random

import matplotlib as mpl
from matplotlib import pyplot as plt, figure

from sklearn.metrics import confusion_matrix

from vna.VNA_defaults import (
    CONFIRM_TEMP_FILE,
    TRAIN_TEST_SEED_VALUE,
    DEFAULT_FIGURE_SIZE,
)
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
    ghz_to_hz,
    get_graph_path,
)
from vna.VNA_enums import (
    MagnitudeOrPhase,
    DfFilterOptions,
    DataFrameCols,
    SParam,
    DfAxis,
    ConfusionMatrixKey,
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
    plot_3d_plots_for_all_gestures_for_sparam,
    confusion_matrix_from_single_result,
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


def plot_confusion_matrix_second_experiment(target_s_param):
    return


def plot_confusion_matrix_original_experiment(target_s_param):
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


def drop_columns_between_frequency(
    data_capture_data_frame, low_freq, high_freq
) -> pd.DataFrame:
    cols_to_drop = list(
        filter(
            lambda x: (low_freq > x) | (x > high_freq),
            data_capture_data_frame.columns[5:],
        )
    )
    return data_capture_data_frame.drop(cols_to_drop, axis=DfAxis.COLUMN)


def process_data_frame_to_3d_plot(
    data_capture_data_frame,
    low_freq,
    high_freq,
    start_time,
    end_time,
    coalesce_duplicates=True,
) -> pd.DataFrame:

    # This is default behaviour and should be done in almost all cases
    if coalesce_duplicates:
        time_series_to_3d_plot = coalesce_duplicate_columns(data_capture_data_frame)

    time_series_to_3d_plot = filter_results_df_between_times(
        time_series_to_3d_plot, start_time, end_time
    )

    time_series_to_3d_plot = drop_columns_between_frequency(
        time_series_to_3d_plot, low_freq, high_freq
    )
    return time_series_to_3d_plot


def plot_3d_plots(
    data_capture_df: pd.DataFrame,
    start_time,
    end_time,
    low_freq,
    high_freq,
    magnitude_or_phase: [MagnitudeOrPhase] = None,
    sparams_to_plot: [SParam] = None,
    title=True,
    file_output_root: Path = None,
    experiment_label=None,
    save_to_file=False,
    figure_size=DEFAULT_FIGURE_SIZE,
):
    if sparams_to_plot is None:
        sparams_to_plot = list(SParam)

    if magnitude_or_phase is None:
        magnitude_or_phase = list(MagnitudeOrPhase)

    time_series_to_3d_plot = process_data_frame_to_3d_plot(
        data_capture_df,
        low_freq,
        high_freq,
        start_time,
        end_time,
        coalesce_duplicates=True,
    )
    sparam: SParam
    measurement: MagnitudeOrPhase
    for sparam in sparams_to_plot:
        for measurement in magnitude_or_phase:
            # file_output_path: Path = file_output_root.joinpath(
            #     measurement.value, sparam.value
            # )
            figures = plot_3d_plots_for_all_gestures_for_sparam(
                time_series_to_3d_plot,
                sparam,
                measurement,
                save_to_file=True,
                experiment_label=experiment_label,
                title=title,
                file_output_path=file_output_root,
                figure_size=figure_size,
            )

            for figure in figures:
                if save_to_file:
                    mpl.pyplot.close(figure)
                else:
                    figure.show()


def get_experiment_label_from_results_object(classifier: dict):
    return [
        ("_").join(i.split("_")[:-1])
        for i in list(classifier["filtered_svm_report"].keys())
    ][1]


if __name__ == "__main__":
    target_sparams_for_3D_time_series = [SParam.S21, SParam.S11, SParam.S31, SParam.S41]
    target_measurements_for_3d_time_series = [
        MagnitudeOrPhase.Magnitude,
        MagnitudeOrPhase.Phase,
    ]
    target_measurements_for_3d_time_series = [MagnitudeOrPhase.Phase]
    target_sparams_for_3D_time_series = [SParam.S11]

    plt.rcParams["font.size"] = 40

    # seed random value for repeatability
    random.seed(TRAIN_TEST_SEED_VALUE)

    CLASSIFIER_RESULTS_PATH = r"C:\Users\2573758S\OneDrive - University of Glasgow\PhD\Experiments\Glove Gesture Experiment\Pickles\Other Related Pkls"
    PKL_RESULTS_FNAME = "classification_results_glove_experiment.pkl"
    EXPERIMENT_NAME = "glove_gesture_experiment_2_201pts_75reps_150M_400M_11ges"
    TOUCHSTONE_FOLDER_PATH = r"C:\Users\2573758S\OneDrive - University of Glasgow\PhD\Experiments\Glove Gesture Experiment\Touchstones\Experiment 2"
    RESULTS_WITHOUT_REPEATS_PKL_FNAME = "glove_experiment_singles_only.pkl"
    FULL_DATA_CAPTURE_A = "glove_gesture_experiment_2_201pts_75reps_150M_400M_11ges.pkl"
    SAVE_TO_FILE = True
    COALESCE_DUPLICATE_COLUMNS_IN_DATAFRAME = True

    OUTPUT_FOLDER_PATH = Path(
        r"C:\Users\2573758S\OneDrive - University of Glasgow\PhD\Experiments\GLove Gesture Experiment 2\Graphs"
    )

    BASE_FOLDER = Path(
        r"C:\Users\2573758S\OneDrive - University of Glasgow\PhD\Experiments\Glove Gesture Experiment\Pickles\Glove Gesture Classification 3 150MHz 350MHz Pkls"
    )

    GESTURE_EXPERIMENT_CAPTURE_DF = (
        r"glove_gesture_experiment_2_201pts_75reps_150M_400M_11ges"
    )
    low_freq = mhz_to_hz(200)
    high_freq = mhz_to_hz(370)
    percent_of_result_to_plot = 20
    start_time = 0
    end_time = 7

    time_series_low_freq = mhz_to_hz(270)
    time_series_high_freq = mhz_to_hz(300)

    confusion_matrix_target_parameter = "gloveExperiment_S21"

    experiment_label = "Glove Experiment"

    # results_df_from_file = combine_classifier_results_dfs(CLASSIFIER_RESULTS_PATH)
    original_results_from_pkls_all_gestures = open_pickled_object(
        Path(
            rf"C:\Users\2573758S\OneDrive - University of Glasgow\PhD\Experiments\Glove Gesture Experiment\Pickles\Other Related Pkls\{PKL_RESULTS_FNAME}"
        )
    )
    # results_df_from_file.to_csv("glove_experiment_csv.csv")
    data_caputre_df = open_pickled_object(BASE_FOLDER.joinpath(FULL_DATA_CAPTURE_A))

    results_df_from_file = open_pickled_object(
        r"C:\Users\2573758S\OneDrive - University of Glasgow\PhD\Experiments\GLove Gesture Experiment 2\glove_experiment_2\gloveExperiment2_40000000MHz_results.pkl"
    )
    classifier = open_pickled_object(
        r"C:\Users\2573758S\OneDrive - University of Glasgow\PhD\Experiments\GLove Gesture Experiment 2\glove_experiment_2\S21_S31_magnitude_0.31_0.39_2025_05_23.pkl"
    )

    # target_frequencies = [
    #     freq
    #     for freq in get_frequency_column_headings_list(data_caputre_df)
    #     if time_series_low_freq <= freq <= time_series_high_freq
    # ]

    confusion_dict = {
        key: val for key, val in classifier.items() if "confusion_matrix" in key
    }
    data = {
        "label": experiment_label,
        "classifier": "svm",
        "full or filtered": "filtered",
        "type": "magnitude",
        "s_param": "S21",
        "low_frequency": "0.31",
        "high_frequency": "0.39",
        "gesture": "accuracy",
    }
    # confusion_matrix_from_single_result(
    #     pd.Series(data),
    #     labels,
    #     confusion_dict,
    #     ConfusionMatrixKey.FILTERED_SVM,
    #     True,
    #     confusion_matrix_key="filtered_svm_confusion_matrix",
    # )
    # run_classification_from_results()

    plot_3d_plots(
        data_caputre_df,
        start_time,
        end_time,
        low_freq,
        high_freq,
        sparams_to_plot=target_sparams_for_3D_time_series,
        magnitude_or_phase=target_measurements_for_3d_time_series,
        file_output_root=OUTPUT_FOLDER_PATH,
        experiment_label=experiment_label,
        title=False,
    )

    # plot_3d_plots(
    #     data_caputre_df,
    #     start_time,
    #     end_time,
    #     low_freq,
    #     high_freq,
    #     sparams_to_plot=target_sparams_for_3D_time_series,
    #     magnitude_or_phase=target_measurements_for_3d_time_series,
    #     file_output_root=OUTPUT_FOLDER_PATH,
    #     experiment_label=experiment_label,
    #     title=True,
    # )

    # os.startfile(OUTPUT_FOLDER_PATH)

    # plot_confusion_matrix_original_experiment(
    #     target_s_param=confusion_matrix_target_parameter
    # )

    # # plot time series plots
    # for target_freq in target_frequencies:
    #     figs = []
    #     for target_s_param in [SParam.S11]:
    #         fig = plot_multiple_gestures_on_time_series(
    #             data_frame=data_caputre_df,
    #             experiment_label=experiment_label,
    #             gestures=list(data_caputre_df["label"].unique()),
    #             target_s_param=target_s_param,
    #             mag_or_phase=MagnitudeOrPhase.Magnitude,
    #             target_frequency=target_freq,
    #             n_random_ids=1,
    #             save_to_file=True,
    #         )
    #         figs.append(fig)
    #
    #     if SAVE_TO_FILE:
    #         for fig in figs:
    #             mpl.pyplot.close(fig)
    #     else:
    #         plt.show()

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

# results_without_repeat = open_pickled_object_in_pickle_folder(
#     RESULTS_WITHOUT_REPEATS_PKL_FNAME
# )
#
# results_without_repeat = convert_magnitude_rows_to_db(results_without_repeat)

# s_param = SParam.S11
# mag_or_phase = MagnitudeOrPhase.Magnitude
#
# single_gesture_for_each_experiment_df = (
#     extract_random_single_gesture_for_each_experiment_to_df(
#         capture_df=results_without_repeat,
#         target_s_param=s_param,
#         mag_or_phase=mag_or_phase,
#     )
# )
#
