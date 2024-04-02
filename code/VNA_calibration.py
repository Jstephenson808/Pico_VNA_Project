import os


class VnaCalibration:
    """
    Holds data related to VNA calibration

    """

    def __init__(
        self,
        calibration_path: os.path,
        number_of_points: int,
        frequncy_range_hz: (int, int),
    ):
        self.calibration_path = calibration_path
        self.number_of_points = number_of_points
        self.low_freq_hz = frequncy_range_hz[0]
        self.high_freq_hz = frequncy_range_hz[1]