from VNA import VNA
from VNA_calibration import VnaCalibration
from VNA_utils import *
from VNA_data import VnaData
from datetime import timedelta
from VNA_enums import *

if __name__ == "__main__":
    #todo make a fn to test lots of cals and decide which gives the best sampling fq
    calibration = VnaCalibration(os.path.join(get_calibration_path(),
                                              "MiniCirc_3dBm_MiniCirc1m_10Mto5G_Rankine506_27Dec23_75kHz_3dBm_401pts.cal"),
                                 401, [mhz_to_hz(10), ghz_to_hz(5)])
    #antenna = "test_1"
    #gestures = ["test_1"]

    antenna = "flex-antenna-watch-140KHz-1001pts-10Mto4G"
    gestures = ["A",
        "B", "C", "I"]

    # calibration = VnaCalibration(os.path.join(get_root_folder_path(), "MiniCirc_3dBm_MiniCirc1m_10Mto6G_101Points_Rankine506_27Dec23_1kHz3dBm.cal"), 101, [10_000_000, 6_000_000_000])
    for gesture in gestures:
        n = 50
        label = f"single_{antenna}_{gesture}"
        input(f"Gesture: {gesture} \nPress Enter To Start")

        # prevent the need to reconnect each time, can keep the cal but need to empty the data frame, then just remeasure

        data = VnaData()

        vna = VNA(calibration, data)

        vna.measure_n_times(timedelta(seconds=2), s_params_output=[param for param in SParam], label=label, print_countdown=True, n_measures=n)

