import os

import numpy as np
import pandas as pd
import random

from VNA_utils import (
    open_pickled_object,
    open_full_results_df,
    get_pickle_path,
    convert_magnitude_cols_to_db,
)
from vna.VNA_enums import (
    SParam2Port,
    MagnitudeOrPhase,
    DfFilterOptions,
    ConfusionMatrixKey,
)
from vna.VNA_utils import (
    open_pickled_object_in_pickle_folder,
    hz_to_mhz,
    mhz_to_hz,
    retype_str_fq_columns_to_int,
    get_none_fq_columns,
    pickle_object,
)
from vna.graphs import (
    plot_fq_time_series,
    plot_multiple_gestures_on_time_series,
    plot_3d_time_series,
    display_confusion_matrix_for_top_n_values,
    bar_graph_accuracy_comparison,
)
from vna.ml_model import (
    fix_measurement_column,
    extract_full_results_to_df,
    extract_confusion_matrix_from_results,
    get_full_results_df_from_classifier_pkls,
)
from vna.scipiCommands import SParam
from vna.single_gesture_classifier import test_classifier_for_all_measured_params
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


def plot_time_series(gestures, target_s_params):

    full_data_frame: pd.DataFrame = open_pickled_object_in_pickle_folder(
        "glove_experiment_results_correct.pkl"
    )
    retype_str_fq_columns_to_int(full_data_frame)
    gesture_repeated = full_data_frame.query(
        "id == 'liquid_metal_glove_6ges_same_gesture_10time_2412181543'"
    )
    gesture_repeated = convert_magnitude_cols_to_db(gesture_repeated)

    gestures = ["A", "B", "C", "1", "2", "3"]

    plot_labels = [
        f"liquid_metal_glove_6ges_same_gesture_10time_{gesture}" for gesture in gestures
    ]
    target_s_params: [SParam] = [SParam.S31, SParam.S21, SParam.S41, SParam.S11]
    # target_s_params: [SParam] = [SParam.S21]

    target_frequency = mhz_to_hz(316)

    for target_s_param in target_s_params:
        plot_multiple_gestures_on_time_series(
            data_frame=gesture_repeated,
            experiment_label="liquid_metal_glove_6ges_same_gesture_10time",
            gestures=gestures,
            target_s_param=target_s_param,
            mag_or_phase=MagnitudeOrPhase.Magnitude,
            target_frequency=target_frequency,
        )


def plot_3d_plots(results_df, s_param, mag_or_phase):

    results_df = convert_magnitude_cols_to_db(results_df)
    experiments = results_df["label"].unique()

    output_df = None
    for experiment in experiments:
        # get all the same label experiments -> this means the same gesture
        same_gesture = results_df[
            (results_df["label"] == experiment)
            & (results_df["s_parameter"] == s_param)
            & (results_df["mag_or_phase"] == mag_or_phase)
        ]

        single_gesture = same_gesture[
            same_gesture["id"] == random.choice(same_gesture["id"].unique())
        ]
        # output will contain one unique gesture capture for each
        output_df: pd.DataFrame = pd.concat(
            [output_df, single_gesture], ignore_index=True
        )

    for val, single_gesture_df in output_df.groupby("label"):

        single_gesture_df = single_gesture_df.reset_index(drop=True)
        chosen_gesture = single_gesture_df["label"][0].split("_")[-1]
        stop_index = 100

        group = single_gesture_df.iloc[:, 4:stop_index].groupby("time")
        plot_3d_time_series(single_gesture_df)


def get_s_param_data(results_df, s_param):
    return results_df[results_df["s_param"] == s_param]


## extract repeats to df
# converter = TouchstoneConverter(
#     touchstone_folder_path=r"C:\Users\2573758S\OneDrive - University of Glasgow\PhD\Experiments\Glove Gesture Experiment\Touchstones\Live Capture Touchstones"
# )
# converter.extract_all_touchstone_data_to_dataframe()
#
# results_df = converter.output_data_frame

# plot_confusion_matrix(target_s_param="gloveExperiment_S21")
# results_df = open_pickled_object_in_pickle_folder(
#     "glove_experiment_results_correct.pkl"
# )
#

# # 3d plots
# results_df = open_pickled_object_in_pickle_folder(
#     "glove_experiment_repeats_only_magnitude.pkl"
# )
#
# low_freq = mhz_to_hz(118)
# high_freq = mhz_to_hz(350)
#
#
# percent_of_result_to_plot = 20
# time_series_to_3d_plot = results_df[(results_df["time"] > 0) & (results_df["time"] < 7)]
# cols_to_drop = list(
#     filter(lambda x: (low_freq > x) | (x > high_freq), results_df.columns[5:])
# )
# time_series_to_3d_plot.drop(cols_to_drop, axis=1, inplace=True)
# #
# #
# for label, df in time_series_to_3d_plot.groupby("id"):
#     plot_3d_plots(df, "S11", "magnitude")
#     plot_3d_plots(df, "S21", "magnitude")

# get new narrow band results

# narrow_band_df = get_full_results_df_from_classifier_pkls(
#     r"D:\James\liquid_dipole_narrow_band"
# )
# pickle_object(
#     narrow_band_df,
#     folder_path=get_pickle_path(),
#     file_name="narrow_band_liquid_dipole_results.pkl",
# )

narrow_band_df = open_pickled_object(
    r"C:\Users\js637s.CAMPUS\PycharmProjects\Pico_VNA_Project\pickles\narrow_band_liquid_dipole_results.pkl"
)
glove_experiment_results = open_pickled_object(
    r"C:\Users\js637s.CAMPUS\PycharmProjects\Pico_VNA_Project\pickles\full_classification_results\glove_experiment.pkl"
)

all_results = pd.concat([glove_experiment_results, narrow_band_df])
all_results["s_param"] = (
    all_results["s_param"]
    .str.replace("gloveExperiment_", "")
    .str.replace("liquid_dipole_18000000_", "")
    .str.replace("liquid_dipole_9000000_", "")
)

all_results["label"] = (
    all_results["label"]
    .str.replace("liquid_metal_glove_6ges_25reps", "Glove Antenna")
    .str.replace("single_liquidAntennaSM3", "Dipole Antenna")
)


accuracy = all_results[(all_results["gesture"] == "accuracy")]
magnitude_only = accuracy[(accuracy["type"] == "magnitude")]

magnitude_s11 = get_s_param_data(magnitude_only, "S11")
magnitude_s21 = get_s_param_data(magnitude_only, "S21")
magnitude_s31 = get_s_param_data(magnitude_only, "S31")
magnitude_s21_s31_s41 = get_s_param_data(magnitude_only, "S21_S31_S41")

# bar_graph_accuracy_comparison(magnitude_s11)
# bar_graph_accuracy_comparison(magnitude_s21)
# bar_graph_accuracy_comparison(magnitude_s21_s31_s41)


# results_without_repeat = open_pickled_object_in_pickle_folder(
#     "glove_experiment_singles_only.pkl"
# )
# results_without_repeat = convert_magnitude_cols_to_db(results_without_repeat)
# experiments = results_without_repeat["label"].unique()
# s_param = "S11"
# mag_or_phase = "magnitude"
#
# output_df = None
# for experiment in experiments:
#     # get all the same label experiments -> this means the same gesture
#     same_gesture = results_without_repeat[
#         (results_without_repeat["label"] == experiment)
#         & (results_without_repeat["s_parameter"] == s_param)
#         & (results_without_repeat["mag_or_phase"] == mag_or_phase)
#     ]
#
#     single_gesture = same_gesture[
#         same_gesture["id"] == random.choice(same_gesture["id"].unique())
#     ]
#     # output will contain one unique gesture capture for each
#     output_df = pd.concat([output_df, single_gesture], ignore_index=True)
#
# single_gesture_df = output_df[
#     output_df["label"] == random.choice(output_df["label"].unique())
# ]
# single_gesture_df = single_gesture_df.reset_index(drop=True)
# chosen_gesture = single_gesture_df["label"][0].split("_")[-1]
# stop_index = 100
#
# group = single_gesture_df.iloc[:, 4:stop_index].groupby("time")

# # plot time series plots
#
# experiment_name = "2412181557_liquid_metal_glove_6ges_25rps"
# gesture_repeated = full_data_frame.query(f"id == '{experiment_name}'")
#
# for target_s_param in target_s_params:
#     plot_multiple_gestures_on_time_series(
#         data_frame=gesture_repeated,
#         experiment_label=experiment_name,
#         gestures=gestures,
#         target_s_param=target_s_param,
#         mag_or_phase=MagnitudeOrPhase.Phase,
#         target_frequency=target_frequency,
#         n_random_ids=5,
#     )

#
# # for plot_label in plot_labels:
# #     for target_s_param in target_s_params:
# #         plot_fq_time_series(
# #             gesture_repeated,
# #             s_parameter=target_s_param,
# #             mag_or_phase=MagnitudeOrPhase.Magnitude,
# #             label=plot_label,
# #             n_random_ids=1,
# #             target_frequency=mhz_to_hz(200),
# #         )


# s_parameter = "S11"
# mag_or_phase = "magnitude"
# label = "single_LIQUID_DIPOLE_SD1_B"
# full_results_df_fname = "sd1_401_75KHz_full_combined_df_2024_07_24.pkl"
#
# full_df = open_full_results_df(full_results_df_fname)
# full_df.columns = list(full_df.columns[:5]) + [int(x) for x in full_df.columns[5:]]
#
# s_param_combinations_list = [["S21", "S31", "S41"], ["S21"], ["S31"], ["S41"]]
#
# # todo need to add svm or dtree label to output dict
# full_results_df = test_classifier_for_all_measured_params(
#     full_df, s_param_combinations_list, DfFilterOptions.BOTH
# )
