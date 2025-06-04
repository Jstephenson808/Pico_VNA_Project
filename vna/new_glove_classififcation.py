import os
from pathlib import Path

import pandas as pd

from vna.VNA_enums import SParam
from vna.VNA_utils import mhz_to_hz, open_pickled_object
from vna.glove_experiment import plot_confusion_matrix_original_experiment
from vna.graphs import display_confusion_matrix_for_top_n_values
from vna.ml_model import (
    extract_confusion_matrix_from_results,
    extract_full_results_to_df,
)

# frequency_ranges = [[mhz_to_hz(200), mhz_to_hz(350)], [mhz_to_hz(250), mhz_to_hz(300)]]
# s_parameter_sets = [
#     [SParam.S21, SParam.S31, SParam.S41],
#     [SParam.S11, SParam.S41],
#     [SParam.S21, SParam.S41],
#     [SParam.S11],
# ]

RESULTS_PATH_STRING = r"C:\Users\2573758S\OneDrive - University of Glasgow\PhD\Experiments\Glove Gesture Experiment 3"
EXPERIMENT_DATA_FRAME_FNAME = "glove_gesture_experiment_2_201pts_75reps_150M_400M_11ges"

results_path: Path = Path(RESULTS_PATH_STRING)

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

extracted_results_all_samples: pd.DataFrame = extract_full_results_to_df(
    classification_output_list_all_samples, extract="report"
)
extracted_results_remove_8: pd.DataFrame = extract_full_results_to_df(
    classification_output_list_remove_8, extract="report"
)

# plot confusion matrix for each of the top results

confusion_matrix_dict_all_samples: pd.DataFrame = extract_confusion_matrix_from_results(
    classification_output_list_all_samples
)
confusion_matrix_dict_remove_8: pd.DataFrame = extract_confusion_matrix_from_results(
    classification_output_list_remove_8
)

display_confusion_matrix_for_top_n_values(
    data_capture_df,
    extracted_results_all_samples,
    confusion_matrix_dict_all_samples,
    n=1,
    convert_to_percent_of_true_labels=True,
)

display_confusion_matrix_for_top_n_values(
    data_capture_df_remove_8,
    extracted_results_remove_8,
    confusion_matrix_dict_remove_8,
    convert_to_percent_of_true_labels=True,
)
