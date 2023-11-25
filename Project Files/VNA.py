import os
from typing import List

import pandas
import win32com.client
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
from enum import Enum


def mhz_to_hz(mhz):
    return mhz * 1000000


class MeasureSParam(Enum):
    S11 = "S11"
    S21 = "S21"
    S11_S21 = "S11+S21"
    ALL = "All"


class SParam(Enum):
    S11 = "S11"
    S21 = "S21"
    S12 = "S12"
    S22 = "S22"


class MeasurementFormat(Enum):
    LOGMAG = "logmag"
    PHASE = "phase"
    REAL = "real"
    IMAG = "imag"
    SWR = "swr"
    GROUP_DELAY = "gd"
    TIME_DOMAIN = "td"


class DataFrameCols(Enum):
    TIME = "time"
    S_PARAMETER = "s_parameter"
    FREQUENCY = "frequency"
    MAGNITUDE = "magnitude"
    PHASE = "phase"


def change_to_output_dir(dir='Figures'):
    os.chdir("..")
    os.chdir(dir)


class VnaCalibration:

    def __init__(self, calibration_path: os.path, number_of_points: int, frequncy_range_hz: (int, int)):
        self.calibration_path = calibration_path
        self.number_of_points = number_of_points
        self.low_freq_hz = frequncy_range_hz[0]
        self.high_freq_hz = frequncy_range_hz[1]


class VnaData:

    def __init__(self):
        self.data_frame = pd.DataFrame(columns=[cols.value for cols in DataFrameCols])
        self.csv_path = None

    def read_df_from_csv(self, path):
        self.data_frame = pd.read_csv(path)
        self.csv_path = path

    def extract_freq_df(self, target_frequency: int, s_param: SParam = None) -> pd.DataFrame:
        if s_param is None:
            df = self.data_frame.loc[(self.data_frame[DataFrameCols.FREQUENCY.value] == target_frequency)]
        else:
            df = self.data_frame.loc[(self.data_frame[DataFrameCols.FREQUENCY.value] == target_frequency)
                                     & (self.data_frame[DataFrameCols.S_PARAMETER] == s_param)]
        return df

    def save_df(self, data_frame: pandas.DataFrame, file_path):
        data_frame.to_csv(file_path)

    def plot_freq_on_axis(self, data_frame, axis: plt.Axes, plot_data: DataFrameCols):
        axis.plot(data_frame[DataFrameCols.TIME.value], data_frame[plot_data.value])

    def single_freq_plotter(self,
                            target_frequency: int,
                            folder,
                            plot_s_param: SParam = None,
                            output_dir='Figures'):
        # how can I get the time out?
        # data_frame['dt'] = calculate_dt(data_frame)
        # data_frame['dt'][0] = 0.0

        # data is a single frame with:
        #  -

        data_frame = self.extract_freq_df(target_frequency, plot_s_param)
        target_frequency_GHz = target_frequency / 1000000000

        fig, ax = plt.subplots()
        self.plot_freq_on_axis(data_frame, ax, DataFrameCols.MAGNITUDE)
        ax.set_ylabel(f"|{plot_s_param.value}|")
        ax.set_xlabel("Time (s)")
        plt.title(f'|{plot_s_param.value}| Over Time at {target_frequency_GHz} GHz')

        change_to_output_dir(output_dir)

        plt.savefig(os.path.join(os.getcwd(),
                    f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}-{target_frequency_GHz}_GHz.png"))

        plt.show()


class VNA:

    def __init__(self,
                 calibration: VnaCalibration,
                 vna_data: VnaData,
                 vna_string="PicoControl2.PicoVNA_2"):
        self.calibration = calibration
        self.vna_object = win32com.client.gencache.EnsureDispatch(vna_string)
        self.output_data = vna_data

    def connect(self):
        print("Connecting VNA")
        search_vna = self.vna_object.FND()
        print('VNA ' + str(search_vna) + ' Loaded')

    def load_cal(self):
        print("Load Calibration")
        ans = self.vna_object.LoadCal(self.calibration.calibration_path)
        print("Result " + str(ans))

    def get_data(self, parameter: SParam, data_format: MeasurementFormat, point=0) -> str:
        return self.vna_object.GetData(parameter, data_format, point)

    def split_data_string(self, data_string: str):

        data_list: list[str] = data_string.split(',')
        frequencies = data_list[::2]
        data = data_list[1::2]
        return frequencies, data

    def data_string_to_df(self,
                          time: timedelta,
                          magnitude_data_string: str,
                          phase_data_string: str,
                          parameter: SParam
                          ) -> pandas.DataFrame:

        frequencies, magnitudes = self.split_data_string(magnitude_data_string)
        frequencies, phase = self.split_data_string(phase_data_string)

        data_dict = {
            DataFrameCols.TIME.value: [time for _ in frequencies],
            DataFrameCols.S_PARAMETER.value: [parameter for _ in frequencies],
            DataFrameCols.FREQUENCY.value: frequencies,
            DataFrameCols.MAGNITUDE.value: magnitudes,
            DataFrameCols.PHASE.value: phase
        }
        return pd.DataFrame(data_dict)

    def generate_output_filename(self, fname: str, s_params_measure: MeasureSParam, run_time: timedelta):
        if fname != "":
            fname += "_"

        return f'{fname}{s_params_measure}_{run_time.seconds}_secs.csv'

    def measure(self,
                run_time: timedelta,
                s_params_measure: MeasureSParam = MeasureSParam.ALL,
                s_param_output: [SParam] = None,
                file_name: str = "",
                output_dir: str = "VNA_Output") -> VnaData:

        if s_param_output is None:
            s_param_output = [SParam.S21]

        change_to_output_dir(output_dir)
        self.output_data.csv_path = self.generate_output_filename(file_name, s_params_measure, run_time)

        self.connect()
        self.load_cal()

        start_time = datetime.now()
        finish_time = start_time + run_time

        measurement_number = 0
        current_time = datetime.now()

        while current_time < finish_time:
            current_time = datetime.now()
            elapsed_time = current_time - start_time

            self.vna_object.Measure(s_params_measure)

            for s_param in s_param_output:
                df = self.data_string_to_df(
                    elapsed_time,
                    self.get_data(s_param, MeasurementFormat.LOGMAG),
                    self.get_data(s_param, MeasurementFormat.PHASE),
                    s_param,
                )

                self.output_data.data_frame = pd.concat([self.output_data.data_frame, df])

            measurement_number += 1
            if measurement_number % 10 == 0:
                print(
                    f"Saving df data index is {measurement_number} running for another {(finish_time - datetime.now())}")
                self.output_data.data_frame.to_csv(self.output_data.csv_path, index=False)

        self.output_data.data_frame.to_csv(self.output_data.csv_path, index=False)

        self.vna_object.CloseVNA()
        print("VNA Closed")
        return self.output_data
