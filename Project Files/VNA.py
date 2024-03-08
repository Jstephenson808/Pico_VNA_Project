import os
from typing import List
import re
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


class NotValidCSVException(Exception):
    def __init__(self, message):
        super().__init__(message)


class VnaCalibration:

    def __init__(self, calibration_path: os.path, number_of_points: int, frequncy_range_hz: (int, int)):
        self.calibration_path = calibration_path
        self.number_of_points = number_of_points
        self.low_freq_hz = frequncy_range_hz[0]
        self.high_freq_hz = frequncy_range_hz[1]


class VnaData:

    @staticmethod
    def freq_string_to_list(string):
        if string.startswith('[') and string.endswith(']'):
            return [int(i) for i in eval(string)]
        else:
            return string

    @staticmethod
    def mag_string_to_list(string):
        if string.startswith('[') and string.endswith(']'):
            return [float(i) for i in eval(string)]
        else:
            return string

    @staticmethod
    def zero_ref_time(data_frame: pd.DataFrame):
        start_time = data_frame['Time'][0]
        data_frame['Time'] = data_frame['Time'].apply(lambda x: (x - start_time).total_seconds())

    @staticmethod
    def string_to_datetime(string):
        return datetime.strptime(string, "%Y_%m_%d_%H_%M_%S.%f")

    def __init__(self):
        self.data_frame = pd.DataFrame(columns=[cols.value for cols in DataFrameCols])
        self.csv_path = None
        self.date_time: datetime = None

    def test_df_columns(self, data_frame: pd.DataFrame):
        assert all(col.value in data_frame.columns for col in DataFrameCols)

    def freq_int_from_ghz_string(self, ghz_string: str):
        return int(float((ghz_string).split('_')[0]) * 1_000_000_000)

    def extract_data_from_old_df(self, old_df: pd.DataFrame, path):
        new_df = pd.DataFrame(columns=[cols.value for cols in DataFrameCols])
        filename = os.path.basename(path)
        match_ghz_fq_csv = re.search(r'(.+)-(\d+\.\d+_GHz)_([A-Za-z\d]+)\.csv', filename)

        if match_ghz_fq_csv:
            self.date_time = datetime.strptime(match_ghz_fq_csv.group(1), "%Y_%m_%d_%H_%M_%S")
            frequency_string = match_ghz_fq_csv.group(2)
            frequency = self.freq_int_from_ghz_string(frequency_string)
            s_param = match_ghz_fq_csv.group(3)
            new_df[DataFrameCols.TIME] = old_df['time']
            new_df[DataFrameCols.MAGNITUDE] = old_df['magnitude (dB)']
            new_df[DataFrameCols.FREQUENCY] = frequency
            new_df[DataFrameCols.S_PARAMETER] = s_param
            return new_df

        if all(col in old_df.columns for col in ['Time', 'Frequency', 'Magnitude (dB)']):
            old_df['Time'] = old_df['Time'].apply(self.string_to_datetime)
            self.date_time = old_df['Time'][0]
            self.zero_ref_time(old_df)
            old_df['Frequency'] = old_df['Frequency'].apply(self.freq_string_to_list)
            old_df['Magnitude (dB)'] = old_df['Magnitude (dB)'].apply(self.mag_string_to_list)
            for index, row in old_df.iterrows():
                temp_df = pd.DataFrame(columns=[cols.value for cols in DataFrameCols])
                temp_df[DataFrameCols.FREQUENCY.value] = row['Frequency']
                temp_df[DataFrameCols.MAGNITUDE.value] = row['Magnitude (dB)']
                temp_df[DataFrameCols.TIME.value] = row['Time']
                new_df = pd.concat([new_df, temp_df])
            return new_df
        raise NotValidCSVException(f"Incorrect CSV format read in with fname {filename} and columns {old_df.columns}")

    def read_df_from_csv(self, path):
        """
        Read in data frame from .csv checks and converts old format files automatically
        or raises exception if not correctly formatted
        :param path: path string to .csv
        :return:
        """
        data_frame = pd.read_csv(path)
        if all(col.value in data_frame for col in DataFrameCols):
            self.data_frame = data_frame
        else:
            self.data_frame = self.extract_data_from_old_df(data_frame, path)

        self.csv_path = path

    def extract_freq_df(self, target_frequency: int, s_param: SParam = None) -> pd.DataFrame:
        """
        Takes in a target frequency and optional sparam, returns a data frame containing only those values and optionally
        only those sparams
        :param target_frequency: Frequency to find
        :param s_param: SParam enum value to search for
        :return: data frame containing only those values
        """
        if s_param is None:
            df = self.data_frame.loc[(self.data_frame[DataFrameCols.FREQUENCY.value] == target_frequency)]
        else:
            df = self.data_frame.loc[(self.data_frame[DataFrameCols.FREQUENCY.value] == target_frequency)
                                     & (self.data_frame[DataFrameCols.S_PARAMETER] == s_param)]
        return df

    def save_df(self, data_frame: pandas.DataFrame, file_path):
        """
        Write data frame to given file path
        :param data_frame: input data frame
        :param file_path: path string to write to
        :return:
        """
        data_frame.to_csv(file_path)

    def plot_freq_on_axis(self, data_frame, axis: plt.Axes, plot_data: DataFrameCols):
        """
        Function which plots the targeted plot_data data type on the y of the supplied axis and
        the value of time on the x axis
        :param data_frame: input data frame
        :param axis: matplotlib axis to plot on
        :param plot_data: The enum datatype to plot
        :return:
        """
        axis.plot(data_frame[DataFrameCols.TIME.value], data_frame[plot_data.value])

    def single_freq_plotter(self,
                            target_frequency: int,
                            output_folder_path = os.path.join(os.path.dirname(__file__), "results", "graphs"),
                            plot_s_param: SParam = None,
                            data_frame_column_to_plot: DataFrameCols = DataFrameCols.MAGNITUDE,
                            ):
        """
        Plots a single frequency from the internal data frame, saves it to the provided folder
        :param target_frequency: frequency in Hz to be plotted
        :param output_folder_path: string for the output folder, defaults to /results/graphs
        :param plot_s_param: optional enum indicating Sparam to be plotted
        :param data_frame_column_to_plot: 
        :return:
        """
        # data is a single frame with:
        #  -

        data_frame = self.extract_freq_df(target_frequency, plot_s_param)
        target_frequency_GHz = target_frequency / 1000000000

        fig, ax = plt.subplots()
        self.plot_freq_on_axis(data_frame, ax, )
        ax.set_ylabel(f"|{plot_s_param.value}|")
        ax.set_xlabel("Time (s)")
        plt.title(f'|{plot_s_param.value}| Over Time at {target_frequency_GHz} GHz')

        plt.savefig(os.path.join(output_folder_path, f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}-{target_frequency_GHz}_GHz.png"))


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
        print(f"VNA {str(search_vna)} Loaded")

    def load_cal(self):
        print("Loading Calibration")
        ans = self.vna_object.LoadCal(self.calibration.calibration_path)
        print(f"Result {ans}")

    def get_data(self, s_parameter: SParam, data_format: MeasurementFormat, point=0) -> str:
        """

        :param s_parameter: S Param data to be returned
        :param data_format: measurement requested
        :param point:
        :return: data string which is ',' separted in the format "freq, measurement_value_at_freq, freq, measurement_value_at_freq"
        """
        return self.vna_object.GetData(s_parameter, data_format, point)

    def split_data_string(self, data_string: str):
        """

        :param data_string: Data string which is ',' separted in the format "freq, measurement_value_at_freq, freq,
                            measurement_value_at_freq" output from self.get_data()
        :return: tuple containing list of frequencies and list of data values
        """
        data_list: list[str] = data_string.split(',')
        frequencies = data_list[::2]
        data = data_list[1::2]
        return frequencies, data

    def vna_data_string_to_df(self,
                              elapsed_time: timedelta,
                              magnitude_data_string: str,
                              phase_data_string: str,
                              s_parameter: SParam
                              ) -> pandas.DataFrame:
        """
        Converts the strings returned by the VNA .get_data method into a data frame 
        with the elapsed time, measured SParam, frequency, mag and phase
        :param elapsed_time: timedelta representing elapsed time when the reading was taken
        :param magnitude_data_string: data string returned by get_data method with magnitude argument
        :param phase_data_string: phase data string returned by get_data method with phase argument
        :param s_parameter: SParam enum value represting the measured Sparam
        :return: pd dataframe formatted correctly to be appended to the data frame in memory
        """
        # todo fix this so you can have phase or mag independently
        frequencies, magnitudes = self.split_data_string(magnitude_data_string)
        frequencies, phase = self.split_data_string(phase_data_string)

        data_dict = {
            DataFrameCols.TIME.value: [elapsed_time for _ in frequencies],
            DataFrameCols.S_PARAMETER.value: [s_parameter for _ in frequencies],
            DataFrameCols.FREQUENCY.value: frequencies,
            DataFrameCols.MAGNITUDE.value: magnitudes,
            DataFrameCols.PHASE.value: phase
        }
        return pd.DataFrame(data_dict)

    def generate_output_filename(self, output_folder: str, s_params_saved: SParam, run_time: timedelta, fname=""):
        """
        Utility function to generate file names
        :param s_params_measure: measured s parameteres
        :param run_time:
        :param fname:
        :return:
        """
        if fname != "":
            fname += "_"

        s_params = ("_").join(s_params_saved.value)
        filename = f'{fname}{s_params}_{run_time.seconds}_secs.csv'
        return os.join(output_folder, 'data', filename)

    # todo what if you only want to measure one of phase or logmag?
    def add_measurement_to_data_frame(self, s_param: SParam, elapsed_time: timedelta):
        """
        Gets current measurement strings (logmag and phase) for the given S param from VNA and converts it to a pd data frame, appending
        this data frame to the output data
        :param s_param: SParam to get the data
        :param elapsed_time: The elaspsed time of the current test (ie the time the data was captured, referenced to 0s)
        :return: the data frame concated on to the current output
        """
        df = self.vna_data_string_to_df(
            elapsed_time,
            self.get_data(s_param, MeasurementFormat.LOGMAG),
            self.get_data(s_param, MeasurementFormat.PHASE),
            s_param,
        )
        return pd.concat([self.output_data.data_frame, df])

    def take_measurement(self, s_params_measure: MeasureSParam, s_params_output: [SParam], elapsed_time: timedelta):
        """
        Takes measurement on the VNA, processes it and appends it to the output_data.data_frame
        df
        :param s_params_measure: The S params for the VNA to measure, using
        :param s_params_output:
        :param elapsed_time:
        """
        # todo what does this return? format?
        self.vna_object.Measure(s_params_measure)

        # todo check how the measurement formats work, where is phase and logmag defined?
        for s_param in s_params_output:
            self.output_data.data_frame = self.add_measurement_to_data_frame(s_param, elapsed_time)

    def measure(self,
                run_time: timedelta,
                s_params_measure: MeasureSParam = MeasureSParam.ALL,
                s_params_output: [SParam] = None,
                file_name: str = "",
                output_dir=os.path.join(os.path.dirname(__file__), 'results')) -> VnaData:

        if s_params_output == None:
            s_params_output = [SParam.S11]

        # todo fix paths
        self.output_data.csv_path = self.generate_output_filename(output_dir, s_params_output, run_time, file_name)
        print(f"Saving to {self.output_data.csv_path}")

        self.connect()
        self.load_cal()

        start_time = datetime.now()
        finish_time = start_time + run_time
        current_time = datetime.now()
        measurement_number = 0

        while current_time < finish_time:
            current_time = datetime.now()
            elapsed_time = current_time - start_time

            self.take_measurement(s_params_measure, s_params_output, elapsed_time)

            measurement_number += 1
            if measurement_number % 10 == 0:
                print(
                    f"Saving df data index is {measurement_number} running for another {(finish_time - datetime.now())}")
                self.output_data.data_frame.to_csv(self.output_data.csv_path, index=False)

        self.output_data.data_frame.to_csv(self.output_data.csv_path, index=False)

        self.vna_object.CloseVNA()
        print("VNA Closed")
        return self.output_data
