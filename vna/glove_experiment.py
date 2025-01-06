import os

import numpy as np
import pandas as pd

from VNA_utils import open_pickled_object, open_full_results_df, get_pickle_path
from vna.VNA_enums import SParam2Port, MagnitudeOrPhase, DfFilterOptions
from vna.VNA_utils import (
    open_pickled_object_in_pickle_folder,
    hz_to_mhz,
    mhz_to_hz,
    retype_str_fq_columns_to_int,
    get_none_fq_columns,
)
from vna.graphs import plot_fq_time_series, plot_multiple_gestures_on_time_series
from vna.ml_model import fix_measurement_column
from vna.scipiCommands import SParam
from vna.single_gesture_classifier import test_classifier_for_all_measured_params
from vna.touchstoneConverter import TouchstoneConverter


# Convert
# path = r"C:\Users\2573758S\OneDrive - University of Glasgow\PhD\Experiments\Glove Gesture Experiment\Touchstones\Live Capture Touchstones"
#
# converter = TouchstoneConverter(touchstone_folder_path=path)
# converter.extract_all_touchstone_data_to_dataframe()


full_data_frame: pd.DataFrame = open_pickled_object_in_pickle_folder(
    "glove_experiment_results.pkl"
)
retype_str_fq_columns_to_int(full_data_frame)
gesture_repeated = full_data_frame.query(
    "id == 'liquid_metal_glove_6ges_same_gesture_10time_2412181543'"
)

gestures = ["A", "B", "C", "1", "2", "3"]

plot_labels = [
    f"liquid_metal_glove_6ges_same_gesture_10time_{gesture}" for gesture in gestures
]
target_s_params: [SParam] = [SParam.S31, SParam.S21, SParam.S41]
# target_s_params: [SParam] = [SParam.S21]

target_frequency = mhz_to_hz(200)

for target_s_param in target_s_params:
    plot_multiple_gestures_on_time_series(
        data_frame=gesture_repeated,
        experiment_label="liquid_metal_glove_6ges_same_gesture_10time",
        gestures=gestures,
        target_s_param=target_s_param,
        mag_or_phase=MagnitudeOrPhase.Magnitude,
        target_frequency=target_frequency,
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

# improve saving of full results so that can happen
# set up all sparams -> permutations
#

s_parameter = "S11"
mag_or_phase = "magnitude"
label = "single_LIQUID_DIPOLE_SD1_B"
full_results_df_fname = "sd1_401_75KHz_full_combined_df_2024_07_24.pkl"

full_df = open_full_results_df(full_results_df_fname)
full_df.columns = list(full_df.columns[:5]) + [int(x) for x in full_df.columns[5:]]

s_param_combinations_list = [["S21", "S31", "S41"], ["S21"], ["S31"], ["S41"]]

# todo need to add svm or dtree label to output dict
full_results_df = test_classifier_for_all_measured_params(
    full_df, s_param_combinations_list, DfFilterOptions.BOTH
)
# combine dfs
full_df_fname = os.listdir(os.path.join(get_pickle_path(), "full_dfs"))[0]
experiment = "watch_small_antenna_1001_140KHz"
full_results_df = combine_results_and_test(os.path.join(get_data_path(), experiment))

# pickle_object(
#     full_results_df, path=os.path.join(get_pickle_path(), "classifier_results"), file_name=f"full_results_17_09_patent_exp"
# )
#
# s11 = full_data_frame.query("id == 'liquid_metal_glove_6ges_same_gesture_10time_2412181543' & s_parameter == 'S11' & label == 'liquid_metal_glove_6ges_same_gesture_10time_1' & mag_or_phase == 'magnitude'")
# full_data_frame.query("")
