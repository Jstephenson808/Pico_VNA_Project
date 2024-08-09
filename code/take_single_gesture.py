import os

from VNA import VNA
from VNA_calibration import VnaCalibration
from VNA_data import VnaData
from VNA_utils import get_calibration_path, ghz_to_hz
from VNA_enums import SParam
# For multi-gesture, ML-related recording - COMMENT FOR SINGLE RECORD
from user_parameters import *
# For single-gesture, single record for a time-period - COMMENT next if unused
#from user_parameters_single_test import *

calibration_path = os.path.join(get_calibration_path(),
                                              CALIBRATION_FNAME)

calibration_folder = os.path.join(get_calibration_path(),
                                              'test')

if __name__ == "__main__":
    calibration = VnaCalibration(calibration_path=calibration_path)

    antenna = ANTENNA_LABEL
    gestures = TEST_GESTURES
    for gesture in gestures:
        n = NUMBER_OF_TESTS
        label = f"single_{antenna}_{gesture}"
        print(f"Label is {label}")
        print(os.path.basename(calibration_path))
        input(f"Gesture: {gesture} \nPress Enter To Start")


        data = VnaData()

        vna = VNA(calibration, data)

        #Edit countdown time between gestures here
        vna.measure_n_times(run_time=TEST_TIME,
                            s_params_output=[param for param in SParam],
                            label=label,
                            print_elapsed_time=True,
                            n_measures=n,
                            save_interval=1000,
                            countdown_seconds=2)
       # data.single_freq_plotter(ghz_to_hz(0.4), plot_s_param=SParam.S11, data_frame_column_to_plot=DataFrameCols.MAGNITUDE)

  #  vna_data = VnaData(r"C:\Users\mww19a\PycharmProjects\Pico_VNA_Project\results\data\single_Test_dipole1_xx\single_Test_dipole1_xx_2024_08_09_14_20_34_S11_S21_S12_S22_10_secs.csv")
  #  vna_data.plot_frequencies()