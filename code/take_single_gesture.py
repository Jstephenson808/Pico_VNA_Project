import os

from VNA import VNA
from VNA_calibration import VnaCalibration
from VNA_data import VnaData
from datetime import timedelta
from VNA_utils import get_calibration_path
from VNA_enums import SParam
from user_parameters import *

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

        vna.measure_n_times(run_time=timedelta(seconds=15),
                            s_params_output=[param for param in SParam],
                            label=label,
                            print_elapsed_time=True,
                            n_measures=n,
                            save_interval=1000)

