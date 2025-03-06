import os

from vna.VNA_enums import DfFilterOptions
from vna.VNA_utils import open_pickled_object, mhz_to_hz, pickle_object, get_pickle_path, retype_str_fq_columns_to_int
from vna.single_gesture_classifier import test_classifier_for_all_measured_params


if __name__ == '__main__':
    # open dipole results
    liquid_dipole_results = open_pickled_object(r'C:\Users\2573758S\PycharmProjects\Pico_VNA_Project\pickles\full_dfs\17_09_patent_exp_combined_df.pkl')

    liquid_dipole_results = retype_str_fq_columns_to_int(liquid_dipole_results)

    # channels are 8.5MHz spaced from
    # 600MHz -> 4GHz

    # same channel spacing as glove experiment
    fq_hops = [mhz_to_hz(i) for i in range(9,27,9)]

    # S params are S_11 -> antenna
    # P2,3,4 are all horns and so are wireless
    s_param_combinations_list = [["S11"], ["S21"],  ["S31"], ["S41"]]
    phase_mag = [DfFilterOptions.MAGNITUDE, DfFilterOptions.PHASE]
    experiment_label = "liquid_dipole"
    for fq_hop in fq_hops:
        label = experiment_label + "_" + str(fq_hop)
        full_results_df = test_classifier_for_all_measured_params(
            liquid_dipole_results, s_param_combinations_list, DfFilterOptions.BOTH, fq_hop=fq_hop, label=label
        )
        # combine dfs
        # full_df_fname = os.listdir(os.path.join(get_pickle_path(), "full_dfs"))[0]
        # experiment = "watch_small_antenna_1001_140KHz"
        # full_results_df = combine_results_and_test(os.path.join(get_data_path(), experiment))

        pickle_object(
            full_results_df, folder_path=os.path.join(get_pickle_path(), "classifier_results"),
            file_name=f"{label}_{fq_hop}MHz_results.pkl"
        )