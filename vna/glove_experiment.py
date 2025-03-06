import os
from datetime import datetime

import pandas as pd

from VNA_utils import open_full_results_df, get_pickle_path, pickle_object
from vna.VNA_enums import DfFilterOptions
from vna.VNA_utils import (
    open_pickled_object_in_pickle_folder,
    mhz_to_hz,
    retype_str_fq_columns_to_int,
)
from vna.single_gesture_classifier import test_classifier_for_all_measured_params

# Convert
# path = r"C:\Users\2573758S\OneDrive - University of Glasgow\PhD\Experiments\Glove Gesture Experiment\Touchstones\Live Capture Touchstones"
#
# converter = TouchstoneConverter(touchstone_folder_path=path)
# converter.extract_all_touchstone_data_to_dataframe()

if __name__ == "__main__":

    # full_data_frame: pd.DataFrame = open_pickled_object_in_pickle_folder(
    #     "glove_experiment_singles_only.pkl"
    # )
    # retype_str_fq_columns_to_int(full_data_frame)
    # gesture_repeated = full_data_frame.query(
    #     "id == 'liquid_metal_glove_6ges_same_gesture_10time_2412181543'"
    # )
    #
    # gestures = ["A", "B", "C", "1", "2", "3"]
    #
    # plot_labels = [
    #     f"liquid_metal_glove_6ges_same_gesture_10time_{gesture}" for gesture in gestures
    # ]
    # target_s_params: [SParam] = [SParam.S31, SParam.S21, SParam.S41]
    # # target_s_params: [SParam] = [SParam.S21]
    #
    # target_frequency = mhz_to_hz(200)
    #
    # for target_s_param in target_s_params:
    #     plot_multiple_gestures_on_time_series(
    #         data_frame=gesture_repeated,
    #         experiment_label="liquid_metal_glove_6ges_same_gesture_10time",
    #         gestures=gestures,
    #         target_s_param=target_s_param,
    #         mag_or_phase=MagnitudeOrPhase.Magnitude,
    #         target_frequency=target_frequency,
    #     )

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


    full_results_df_fname = "glove_experiment_singles_only.pkl"
    label = "gloveExperiment"
    full_df = open_full_results_df(full_results_df_fname)
    full_df.columns = list(full_df.columns[:5]) + [int(x) for x in full_df.columns[5:]]

    s_param_combinations_list = [["S11"], ["S21"],  ["S21", "S11"]]
    phase_mag = [DfFilterOptions.MAGNITUDE, DfFilterOptions.PHASE]
    fq_hops = [mhz_to_hz(i) for i in range(10,21,4)]

    for fq_hop in fq_hops:
        full_results_df = test_classifier_for_all_measured_params(
            full_df, s_param_combinations_list, DfFilterOptions.BOTH, fq_hop=fq_hop, label=label
        )
        # combine dfs
        # full_df_fname = os.listdir(os.path.join(get_pickle_path(), "full_dfs"))[0]
        # experiment = "watch_small_antenna_1001_140KHz"
        # full_results_df = combine_results_and_test(os.path.join(get_data_path(), experiment))

        pickle_object(
            full_results_df, folder_path=os.path.join(get_pickle_path(), "classifier_results"), file_name=f"{label}_{fq_hop}MHz_results.pkl"
        )
    #
    # s11 = full_data_frame.query("id == 'liquid_metal_glove_6ges_same_gesture_10time_2412181543' & s_parameter == 'S11' & label == 'liquid_metal_glove_6ges_same_gesture_10time_1' & mag_or_phase == 'magnitude'")
    # full_data_frame.query("")

def id_is_before_time(date_time:datetime, id:str) -> bool:
    test_time = get_datetime_from_id(id)
    return test_time < date_time

def get_datetime_from_id(id_string:str)->datetime:
    just_time_stamp = ('_').join(id_string.split('_')[-6:])
    return datetime.strptime(just_time_stamp, "%Y_%m_%d_%H_%M_%S")