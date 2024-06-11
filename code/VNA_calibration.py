import os
import re
from VNA_utils import get_calibration_path
from VNA_exceptions import NotValidCalibrationFileException

class VnaCalibration:
    """
    Holds data related to VNA calibration

    """

    @staticmethod
    def validate_line(line, pattern):
        if not re.match(pattern, line):
            raise NotValidCalibrationFileException(f"Not a valid calibration: {line}")


    def __init__(
        self,
        calibration_path: os.path
    ):
        self.calibration_path = calibration_path
        self.number_of_points = None
        self.low_freq_hz = None
        self.high_freq_hz = None
        self.fq_hop = None
        self.verify_file_is_cal()
        self.extract_npoints_fq_range()

    def verify_file_is_cal(self):
        sweep_plan_pattern = r'^(\d+,\d+,\d+,\d+,\d+,\d+)$'
        with open(self.calibration_path, 'r') as calibration_txt:
            lines = calibration_txt.readlines()
            self.validate_line(lines[2], sweep_plan_pattern)

    def extract_npoints_fq_range(self):
        with open(self.calibration_path, 'r') as calibration_txt:
            lines = calibration_txt.readlines()
            sweep_plan_list = lines[2].strip().split(',')
            self.number_of_points = sweep_plan_list[0]
            self.low_freq_hz = sweep_plan_list[3]
            self.high_freq_hz = sweep_plan_list[4]
            self.fq_hop = sweep_plan_list[5]


if __name__ == '__main__':
    calibration_path = os.path.join(get_calibration_path(),
                                    "MiniCirc_3dBm_MiniCirc1m_10Mto5G_Rankine506_27Dec23_75kHz_3dBm_401pts.cal")

    calib = VnaCalibration(calibration_path)