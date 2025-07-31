import matplotlib as mpl
from pathlib import Path

import pandas as pd

from vna.VNA_enums import SParam, MagnitudeOrPhase, ConfusionMatrixKey
from vna.VNA_utils import mhz_to_hz, open_pickled_object, pickle_object
from vna.glove_experiment import (
    plot_confusion_matrix_original_experiment,
    plot_3d_plots,
)
from vna.graphs import (
    display_confusion_matrix_for_top_n_values,
    set_graph_svg_text_to_text,
    confusion_matrix_from_single_result,
)
from vna.ml_model import (
    extract_confusion_matrix_from_results,
    extract_full_results_to_df,
)


def extract_remove_8_data(results_path: Path, accuracy_only=True) -> pd.DataFrame:
    classification_output_list_remove_8: [Path] = list(
        results_path.glob("classifiers/remove_8/*.pkl")
    )
    extracted_results_remove_8: pd.DataFrame = extract_full_results_to_df(
        classification_output_list_remove_8, extract="report"
    )
    if accuracy_only:
        extracted_results_remove_8 = extracted_results_remove_8[
            extracted_results_remove_8["gesture"] == "accuracy"
        ]
    return extracted_results_remove_8


# frequency_ranges = [[mhz_to_hz(200), mhz_to_hz(350)], [mhz_to_hz(250), mhz_to_hz(300)]]
# s_parameter_sets = [
#     [SParam.S21, SParam.S31, SParam.S41],
#     [SParam.S11, SParam.S41],
#     [SParam.S21, SParam.S41],
#     [SParam.S11],
# ]
if __name__ == "__main__":

    mpl.rcParams = set_graph_svg_text_to_text(mpl.rcParams)
    font_size = 20

    # This updates everything to use your desired size
    rc = {
        "font.size": font_size,  # Base font size
        "axes.titlesize": font_size,  # Title
        "axes.labelsize": font_size,  # Axis labels
        "xtick.labelsize": font_size,  # X tick labels
        "ytick.labelsize": font_size,  # Y tick labels
        "legend.fontsize": font_size,  # Legend
        "legend.title_fontsize": font_size,  # Legend title
    }
    mpl.rcParams.update(rc)

    # final data folder path
    EXPERIMENT_PKL_FOLDER_STRING = r"C:\Users\2573758S\OneDrive - University of Glasgow\PhD\Experiments\Glove Gesture Experiment\Pickles\Glove Gesture Classification Paper Data"

    # origin results folder
    RESULTS_PATH_STRING = r"C:\Users\2573758S\OneDrive - University of Glasgow\PhD\Experiments\Glove Gesture Experiment 3"

    # results pkl filename
    CLASSIFICATION_WITH_ALL_GESTURES_RESULTS_FNAME = (
        "glove_gesture_experiment_150M_300M_classification_results_all_gestures"
    )
    CLASSIFICATION_REMOVE_GESTURE_8_RESULTS_FNAME = (
        "glove_gesture_experiment_150M_300M_classification_results_remove_8"
    )

    # Confusion Matrix
    CONFUSION_MATRIX_ALL_SAMPLES_FILENAME = (
        "confusion_matrix_dict_glove_experiment_2_all_samples"
    )
    CONFUSION_MATRIX_REMOVE_8_FILENAME = (
        "confusion_matrix_dict_glove_experiment_2_remove_8"
    )

    EXPERIMENT_DATA_FRAME_FNAME = (
        "glove_gesture_experiment_2_201pts_75reps_150M_400M_11ges"
    )

    PLOT_CONFUSION_MATRIX_REMOVE_8 = False
    PLOT_CONFUSION_MATRIX_ALL_SAMPLES = False
    N_SAMPLES_TO_PLOT = 1

    results_path: Path = Path(RESULTS_PATH_STRING)
    experiment_pkl_folder_path = Path(EXPERIMENT_PKL_FOLDER_STRING)

    all_pkl_files_paths: [Path] = list(results_path.glob("**/*.pkl"))

    reclassificaiton_results: pd.DataFrame = pd.concat(
        [
            open_pickled_object(path)
            for path in all_pkl_files_paths
            if "gloveExperiment_reclassifcation" in str(path)
        ]
    )

    data_capture_df: pd.DataFrame = pd.concat(
        [
            open_pickled_object(path)
            for path in all_pkl_files_paths
            if EXPERIMENT_DATA_FRAME_FNAME in str(path)
        ]
    )

    data_capture_df_remove_8: pd.DataFrame = data_capture_df[
        ~data_capture_df["label"].str.endswith("8")
    ]

    classification_output_list_all_samples: [Path] = list(
        results_path.glob("classifiers/all_samples/*.pkl")
    )

    classification_output_list_remove_8: [Path] = list(
        results_path.glob("classifiers/remove_8/*.pkl")
    )

    extracted_results_all_samples: pd.DataFrame = open_pickled_object(
        experiment_pkl_folder_path.joinpath(
            CLASSIFICATION_WITH_ALL_GESTURES_RESULTS_FNAME
        )
    )

    extracted_results_remove_8: pd.DataFrame = open_pickled_object(
        r"C:\Users\2573758S\OneDrive - University of Glasgow\PhD\Experiments\Glove Gesture Experiment\Pickles\Glove Gesture Classification 3 150MHz 350MHz Pkls\classifier_results\extracted_results_glove_experiment_remove8_accuracy_filtered.pkl"
    )

    confusion_matrix_dict_all_samples: dict[str, dict] = open_pickled_object(
        experiment_pkl_folder_path.joinpath(CONFUSION_MATRIX_ALL_SAMPLES_FILENAME)
    )
    confusion_matrix_dict_remove_8: dict[str, dict] = open_pickled_object(
        experiment_pkl_folder_path.joinpath(CONFUSION_MATRIX_REMOVE_8_FILENAME)
    )

    # plot confusion matrix for each of the top results

    if PLOT_CONFUSION_MATRIX_ALL_SAMPLES:

        display_confusion_matrix_for_top_n_values(
            data_capture_df,
            extracted_results_all_samples,
            confusion_matrix_dict_all_samples,
            n=N_SAMPLES_TO_PLOT,
            convert_to_percent_of_true_labels=True,
        )

    if PLOT_CONFUSION_MATRIX_REMOVE_8:
        display_confusion_matrix_for_top_n_values(
            data_capture_df_remove_8,
            extracted_results_remove_8,
            confusion_matrix_dict_remove_8,
            n=N_SAMPLES_TO_PLOT,
            convert_to_percent_of_true_labels=True,
        )

    target_sparams_for_3D_time_series = [SParam.S21, SParam.S11, SParam.S31, SParam.S41]
    target_measurements_for_3d_time_series = [
        MagnitudeOrPhase.Magnitude,
        MagnitudeOrPhase.Phase,
    ]
    # target_measurements_for_3d_time_series = [MagnitudeOrPhase.Phase]
    # target_sparams_for_3D_time_series = [SParam.S11]

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
    labels = ["1", "2", "3", "A", "B", "C", "I", "ILY", "L", "Y"]
    for label, confusion_matrix in confusion_matrix_dict_remove_8.items():
        label_list = label.split("_")
        data["s_param"] = label_list[0]
        data["type"] = label_list[1]
        data["low_frequency"] = label_list[2]
        data["high_frequency"] = label_list[3]
        confusion_matrix_from_single_result(
            pd.Series(data),
            labels,
            confusion_matrix,
            ConfusionMatrixKey.FILTERED_SVM,
            convert_to_percent_of_true_labels=True,
            confusion_matrix_key="filtered_svm_confusion_matrix",
            title=True,
            title_value=label,
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
    #     title=False,
    #     save_to_file=SAVE_TO_FILE,
    #     figure_size=(20, 10),
    # )
