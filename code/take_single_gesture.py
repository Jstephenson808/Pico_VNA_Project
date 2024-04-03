from VNA import VNA
from VNA_calibration import VnaCalibration
from VNA_utils import *
from VNA_data import VnaData
from datetime import timedelta
from VNA_enums import *

if __name__ == "__main__":
    #todo make a fn to test lots of cals and decide which gives the best sampling fq
    calibration = VnaCalibration(os.path.join(get_calibration_path(),
                                              "MiniCirc_3dBm_MiniCirc1m_10Mto5G_Rankine506_27Dec23_140kHz_3dBm_1001pts.cal"),
                                 201, [mhz_to_hz(10), ghz_to_hz(5)])

    antenna = "watchLargeAntennaL-140KHz-1001pts"
    #antenna = "test"
    gestures = ["A", "B", "C", "I", "L", "Y", "1", "2", "3", "8", "I Love You"]

    #gestures = ["test"]
    # calibration = VnaCalibration(os.path.join(get_root_folder_path(), "MiniCirc_3dBm_MiniCirc1m_10Mto6G_101Points_Rankine506_27Dec23_1kHz3dBm.cal"), 101, [10_000_000, 6_000_000_000])
    for gesture in gestures:
        n = 50
        label = f"single_{antenna}_{gesture}"
        input(f"Gesture: {gesture} \nPress Enter To Start")
        while (n > 0):
            print(f"n = {n}")
            n -= 1
            data = VnaData()

            vna = VNA(calibration, data)

            vna.measure(timedelta(seconds=2), s_params_output=[param for param in SParam], label=label, print_countdown=True)

