from abc import ABC, abstractmethod
from datetime import timedelta
import pandas as pd
from pathlib import Path

from .s_parameter_data import SParameterData
from .PicoVNA2 import PicoVNA2
from .VNA_data import VnaData
from .VNA_calibration import VnaCalibration
from .VNA_enums import MeasureSParam, TwoPortSParams


class DataSource(ABC):
    """
    Abstract base class for a data source.
    """

    @abstractmethod
    def read(self, experiment_data) -> SParameterData:
        """
        Read data from the source and return it as an SParameterData object.
        """
        pass


class PicoVNA2DataSource(DataSource):
    """
    A data source that captures live data from the PicoVNA.
    """

    def __init__(self, calibration_path: str, run_time: timedelta, label: str):
        self.calibration_path = calibration_path
        self.run_time = run_time
        self.label = label

    def read(self, experiment_data) -> SParameterData:
        print("Reading from VNA DataSource...")
        calibration = VnaCalibration(self.calibration_path)
        vna_data = VnaData(s_params_to_save=[TwoPortSParams.S21])
        vna = PicoVNA2(calibration=calibration, vna_data=vna_data)

        vna.connect()
        vna.load_cal()

        vna.take_measurement(
            s_params_measure=MeasureSParam.S21,
            s_params_output=[TwoPortSParams.S21],
            elapsed_time=timedelta(seconds=0),
            label=self.label,
            id=experiment_data.experiment_id,
        )

        vna.close_connection()

        df = vna_data.dict_list_to_df()
        return SParameterData(df)


class FileDataSource(DataSource):
    """
    A data source that reads data from a file.
    """

    def __init__(self, file_path: Path, label: str):
        self.file_path = file_path
        self.label = label

    def read(self, experiment_data) -> SParameterData:
        print(f"Reading from File DataSource: {self.file_path}")
        return SParameterData.open_full_results_df(
            label=self.label, path=self.file_path
        )
