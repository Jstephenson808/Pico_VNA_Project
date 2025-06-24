import os
from datetime import datetime
from pathlib import Path

from pynput import keyboard
import threading

from VNA import VNA
from VNA_calibration import VnaCalibration
from VNA_data import VnaData
from datetime import timedelta
from VNA_utils import get_calibration_path, ghz_to_hz
from VNA_enums import SParam2Port, DateFormats
from vna.VNA_enums import DataFrameCols
from vna.VNA_utils import get_labels_path

#
# TIMESTAMPS_FILENAME = (
#     f"{datetime.now().strftime(DateFormats.CURRENT.value)}_timestamps.txt"
# )
# TIMESTAMPS_FILE_OUTPUT_PATH = Path(get_labels_path(), TIMESTAMPS_FILENAME)


from user_parameters import *

calibration_path = os.path.join(get_calibration_path(), CALIBRATION_FNAME)

calibration_folder = os.path.join(get_calibration_path(), "test")

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

        data = VnaData(
            path=r"C:\Users\js637s.CAMPUS\PycharmProjects\Pico_VNA_Project\results\data\single_xiao soil antenna attempt two unrecycled_water_addition\single_xiao soil antenna attempt two unrecycled_water_addition_2025_06_23_17_20_07_S11_S12_S22_S21_1200_secs.csv"
        )

        # data = VnaData()
        #
        # vna = VNA(calibration, data)
        #
        # vna.measure_n_times(
        #     run_time=TEST_TIME,
        #     s_params_output=[param for param in SParam2Port],
        #     label=label,
        #     print_elapsed_time=True,
        #     n_measures=n,
        #     save_interval=SAVE_INTERVAL,
        # )
        for param in SParam2Port:
            data.single_freq_plotter(
                ghz_to_hz(1),
                plot_s_param=param,
                data_frame_column_to_plot=DataFrameCols.MAGNITUDE,
            )
    # data.single_freq_plotter(ghz_to_hz(0.4), plot_s_param=SParam.S11, data_frame_column_to_plot=DataFrameCols.MAGNITUDE)

#  vna_data = VnaData(r"C:\Users\mww19a\PycharmProjects\Pico_VNA_Project\results\data\single_Test_dipole1_xx\single_Test_dipole1_xx_2024_08_09_14_20_34_S11_S21_S12_S22_10_secs.csv")
#  vna_data.plot_frequencies()
