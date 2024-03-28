import os
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

os.environ['OMP_NUM_THREADS'] = "1"
os.environ['MKL_NUM_THREADS'] = "1"
os.environ['OPENBLAS_NUM_THREADS'] = "1"
import win32com.client
from tsfresh import extract_features, select_features
from enum import Enum
from matplotlib import pyplot as plt
from datetime import datetime, timedelta
import ast
from typing import List
import re
import numpy as np
import pandas as pd
import matplotlib
from tsfresh.utilities.dataframe_functions import impute

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
matplotlib.use("TkAgg")
from time import time
import pickle


ROOT_FOLDER = "Pico_VNA_Project"
DATA_FOLDER = os.path.join("results", "data")

class NotValidCSVException(Exception):
    def __init__(self, message):
        super().__init__(message)


class NotValidSParamException(Exception):
    def __init__(self, message):
        super().__init__(message)


class FileNotInCorrectFolder(Exception):
    def __init__(self, message):
        super().__init__(message)


class VNAError(Exception):
    def __init__(self, message):
        super().__init__(message)


class Movements(Enum):
    BEND = "bend"


class DateFormats(Enum):
    CURRENT = "%Y_%m_%d_%H_%M_%S"
    ORIGINAL = "%Y_%m_%d_%H_%M_%S.%f"
    DATE_FOLDER = "%Y_%m_%d"


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
    LABEL = "label"
    ID = "id"



def timer_func(func):
    # This function shows the execution time of
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f"Function {func.__name__!r} executed in {(t2-t1):.4f}s")
        return result

    return wrap_func


def mhz_to_hz(mhz):
    """
    utility function to convert mhz to hz
    :param mhz: MHz value
    :return: value in Hz
    """
    return mhz * 1_000_000


def hz_to_mhz(hz):
    """
    utility function to convert hz to Mhz
    :param hz: Hz value
    :return: value in MHz
    """
    return hz / 1_000_000


def ghz_to_hz(ghz):
    """
    utility function to convert GHz to Hz
    :param ghz: GHz value
    :return: value in Hz
    """
    return ghz * 1_000_000_000


def hz_to_ghz(hz):
    """
    utility function to convert Hz to GHz
    :param hz: Hz value
    :return: value in GHz
    """
    return hz / 1_000_000_000


def get_root_folder_path():
    """
    utility function to get the root folder of the project
    assumes file is running in original folder, this can cause issues
    if the root folder is moved
    :return: path to the root folder
    """
    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.basename(path) != ROOT_FOLDER:
        raise FileNotInCorrectFolder(
            f"File isn't in correct folder for this function to work,"
            f" move into a folder below root dir, currently in {__file__}"
        )
    return path

def get_data_path():
    return os.path.join(get_root_folder_path(), DATA_FOLDER)

def get_pickle_path():
    return os.path.join(get_root_folder_path(), "pickels")

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


class VnaData:
    """
    Class to hold data produced by the VNA and methods for plotting, can be initialised
    with a path to a .csv of just a dataframe value. Provides methods for extracting individual
    frequencies and plotting them, can save these files and plots.
    """

    @staticmethod
    def test_file_name(filename) -> bool:
        """
        Tests file name to establish if it's in the correct format
        :param filename: filename from command line
        :return: bool indicating if the fname is formatted correctly
        """
        pattern = r"^[\w\-.]+$"

        regex = re.compile(pattern)

        return regex.match(filename) is not None

    @staticmethod
    def freq_string_to_list(string):
        """
        converts legacy frequency list string to a list of ints
        :param string: string read in from .csv
        :return:
        """
        if string.startswith("[") and string.endswith("]"):
            return [int(i) for i in ast.literal_eval(string)]

        return string

    @staticmethod
    def mag_string_to_list(string):
        """
        converts legacy magnitude list string to a list of floats
        :param string:
        :return:
        """
        if string.startswith("[") and string.endswith("]"):
            return [float(i) for i in ast.literal_eval(string)]
        return string

    @staticmethod
    def zero_ref_time(data_frame: pd.DataFrame):
        """
        References the time in data frame to first value
        :param data_frame: data frame
        :return: None
        """
        start_time = data_frame[DataFrameCols.TIME.value][0]
        data_frame[DataFrameCols.TIME.value] = data_frame[
            DataFrameCols.TIME.value
        ].apply(lambda x: x - start_time)

    @staticmethod
    def string_to_datetime(
        string, date_format: DateFormats = DateFormats.ORIGINAL.value
    ) -> datetime:
        """
        Converts a string containing a time into a datetime object
        :param string:
        :param date_format: enum containing the format string for datetime
        :return:
        """
        return datetime.strptime(string, date_format)

    @staticmethod
    def freq_int_from_ghz_string(ghz_string: str) -> int:
        """
        Converts string in the format "_._GHz" to an integer Hz value
        :param ghz_string: GHz string in format "_._GHz"
        :return: frequency in Hz
        """
        return int(float(ghz_string.split("_")[0]) * 1_000_000_000)

    @staticmethod
    def read_df_from_csv(path) -> (pd.DataFrame, datetime):
        """
        Read in data frame from .csv checks and converts old format files automatically
        or raises exception if not correctly formatted
        :param path: path string to .csv
        :return: tuple containing data frame and datetime of when data was taken
        """
        data_frame = pd.read_csv(path)
        # if all the columns in this enum are present then .csv was in the right format
        if all(col.value in data_frame for col in DataFrameCols):
            # pattern matches date format in
            pattern = re.compile(r"(\d\d\d\d_\d\d_\d\d_\d\d_\d\d_\d\d)")
            date_time = re.search(pattern, os.path.basename(path)).group(1)
            return data_frame, datetime.strptime(date_time, DateFormats.CURRENT.value)
        else:
            return VnaData.extract_data_from_old_df(data_frame, path)

    @staticmethod
    def extract_data_from_old_df(
        old_df: pd.DataFrame, path
    ) -> (pd.DataFrame, datetime):
        """
        If a dataframe is read in which is not of the current format,
        this function is called to convert the old format into the new one,
        if it is not of a known format it will throw a NotValidCsv exception

        :param old_df: The old data frame which is not in the current format
        :param path: The path to the read in .csv
        :return: (pd.DataFrame, datetime)
        """
        new_df = pd.DataFrame(columns=[cols.value for cols in DataFrameCols])
        filename = os.path.basename(path)
        match_ghz_fq_csv = re.search(
            r"(.+)-(\d+\.\d+_GHz)_([A-Za-z\d]+)\.csv", filename
        )

        if match_ghz_fq_csv:
            date_time = datetime.strptime(
                match_ghz_fq_csv.group(1), DateFormats.CURRENT.value
            )
            frequency_string = match_ghz_fq_csv.group(2)
            frequency = VnaData.freq_int_from_ghz_string(frequency_string)
            s_param = match_ghz_fq_csv.group(3)
            new_df[DataFrameCols.TIME.value] = old_df["time"]
            new_df[DataFrameCols.MAGNITUDE.value] = old_df["magnitude (dB)"]
            new_df[DataFrameCols.FREQUENCY.value] = frequency
            new_df[DataFrameCols.S_PARAMETER.value] = s_param
            return new_df, date_time

        if all(
            col in old_df.columns for col in ["Time", "Frequency", "Magnitude (dB)"]
        ):
            s_param = re.search(r"(S\d\d)", filename).group(1)
            old_df["Time"] = old_df["Time"].apply(VnaData.string_to_datetime)
            date_time = old_df["Time"][0]
            VnaData.zero_ref_time(old_df)
            old_df["Frequency"] = old_df["Frequency"].apply(VnaData.freq_string_to_list)
            old_df["Magnitude (dB)"] = old_df["Magnitude (dB)"].apply(
                VnaData.mag_string_to_list
            )
            for index, row in old_df.iterrows():
                temp_df = pd.DataFrame(columns=[cols.value for cols in DataFrameCols])
                temp_df[DataFrameCols.FREQUENCY.value] = row["Frequency"]
                temp_df[DataFrameCols.MAGNITUDE.value] = row["Magnitude (dB)"]
                temp_df[DataFrameCols.TIME.value] = row["Time"]
                temp_df[DataFrameCols.S_PARAMETER.value] = s_param
                new_df = pd.concat([new_df, temp_df], ignore_index=True)
            return new_df, date_time
        raise NotValidCSVException(
            f"Incorrect CSV format read in with fname {filename} "
            f"and columns {old_df.columns}"
        )

    def __init__(self, path=None, data_frame=None, date_time=None):
        self.data_frame: pd.DataFrame = data_frame
        self.date_time: datetime = date_time
        self.csv_path = path
        if path is not None:
            self.init_df_date_time()

    def init_df_date_time(self):
        self.data_frame, self.date_time = VnaData.read_df_from_csv(self.csv_path)

    def get_first_index_of_time(self, target_time, target_magnitude=None):
        if target_magnitude is None:
            filtered_indexes = self.data_frame.index[
                self.data_frame[DataFrameCols.TIME.value] > target_time
            ]
        else:
            filtered_indexes = self.data_frame.index[
                (self.data_frame[DataFrameCols.TIME.value] > target_time)
                & self.data_frame[DataFrameCols.MAGNITUDE.value]
                > target_magnitude
            ]

        if len(filtered_indexes) == 0:
            raise IndexError(f"Target time out of bounds {target_time}")
        else:
            return filtered_indexes[0]

    def split_data_frame(self, n_slices, start_time=0, start_magnitude=None):
        start_index = self.get_first_index_of_time(start_time, start_magnitude)
        split_data_frames = np.array_split(self.data_frame[start_index:], n_slices)
        index_reset_df = []
        for data_frame in split_data_frames:
            index_reset_df.append(data_frame.reset_index())
        return index_reset_df

    def test_df_columns(self, data_frame: pd.DataFrame):
        assert all(col.value in data_frame.columns for col in DataFrameCols)

    def extract_freq_df(
        self, target_frequency: int, s_param: SParam = None
    ) -> pd.DataFrame:
        """
        Takes in a target frequency and optional sparam,
        returns a data frame containing only those values
        and optionally only those sparams

        :param target_frequency: Frequency to find
        :param s_param: SParam enum value to search for
        :return: data frame containing only those values
        """
        if s_param is None:
            df = self.data_frame.loc[
                (self.data_frame[DataFrameCols.FREQUENCY.value] == target_frequency)
            ]
        else:
            df = self.data_frame.loc[
                (self.data_frame[DataFrameCols.FREQUENCY.value] == target_frequency)
                & (self.data_frame[DataFrameCols.S_PARAMETER.value] == s_param.value)
            ]
        return df.sort_values(by=[DataFrameCols.TIME.value])

    def save_df(self, file_path=os.path.join("../Project Files/results", "data")):
        """
        Write data frame to given file path
        :param data_frame: input data frame
        :param file_path: path string to write to
        :return:
        """
        os.makedirs(file_path, exist_ok=True)
        self.data_frame.to_csv(file_path, index=False)

    def plot_frequencies(self, freq_list: [int], output_folder_path=os.path.join(
                            get_root_folder_path(),
                            "results",
                            "graph",
                            datetime.now().date().strftime(DateFormats.DATE_FOLDER.value)),
                            plot_s_param: SParam = None,
                            data_frame_column_to_plot: DataFrameCols = DataFrameCols.MAGNITUDE,
                            save_to_file=True,):

        fig, ax = plt.subplots()
        ax.set_ylabel(f"|{plot_s_param.value}|")
        ax.set_xlabel("Time (s)")
        plt.title(f"{plot_s_param.value.title()} Over Time")
        for target_frequency in freq_list:
            target_frequency = self.find_nearest_frequency(
                self.data_frame[DataFrameCols.FREQUENCY.value], target_frequency
            )
            data_frame = self.extract_freq_df(target_frequency, plot_s_param)
            target_frequency_GHz = hz_to_ghz(target_frequency)
            self.plot_freq_on_axis(
                data_frame, ax, data_frame_column_to_plot, label=target_frequency_GHz
            )
        plt.legend()
        if save_to_file:
            os.makedirs(output_folder_path, exist_ok=True)
            plt.savefig(
                os.path.join(
                    output_folder_path,
                    f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}_{freq_list[0]}-{freq_list[-1]}.svg",
                ),
                format="svg",
            )
        plt.show()

    def plot_freq_on_axis(
        self, data_frame, axis: plt.Axes, plot_data: DataFrameCols, label=None
    ):
        """
        Function which plots the targeted plot_data data type on the y of the supplied axis and
        the value of time on the x axis
        :param data_frame: input data frame
        :param axis: matplotlib axis to plot on
        :param plot_data: The enum datatype to plot
        :return:
        """
        if label:
            axis.plot(
                data_frame[DataFrameCols.TIME.value],
                data_frame[plot_data.value],
                label=label,
            )
        else:
            axis.plot(data_frame[DataFrameCols.TIME.value], data_frame[plot_data.value])

    def find_nearest_frequency(self, frequency_series: pd.Series, target_frequency):
        array = np.asarray(frequency_series)
        idx = (np.abs(array - target_frequency)).argmin()
        return array[idx]

    def validate_s_param(self, plot_s_param: SParam) -> bool:
        if (plot_s_param is None) or (plot_s_param not in SParam):
            raise NotValidSParamException(f"{plot_s_param} is not valid")
        return True

    def handle_none_param(self, plot_s_param: None) -> SParam | None:
        plot_s_param_string = self.data_frame[DataFrameCols.S_PARAMETER.value].values[0]
        for param_enum in SParam:
            if param_enum.value == plot_s_param_string:
                plot_s_param = param_enum
                break
        return plot_s_param

    def single_freq_plotter(
        self,
        target_frequency: int,
        output_folder_path=os.path.join(
            get_root_folder_path(),
            "results",
            "graph",
            datetime.now().date().strftime(DateFormats.DATE_FOLDER.value),
        ),
        plot_s_param: SParam = None,
        data_frame_column_to_plot: DataFrameCols = DataFrameCols.MAGNITUDE,
        save_to_file=True,
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
        # if no sparam is given just pick the first value of SParam
        if plot_s_param == None:
            plot_s_param = self.handle_none_param(plot_s_param)

        self.validate_s_param(plot_s_param)

        target_frequency = self.find_nearest_frequency(
            self.data_frame[DataFrameCols.FREQUENCY.value], target_frequency
        )
        data_frame = self.extract_freq_df(target_frequency, plot_s_param)
        target_frequency_GHz = hz_to_ghz(target_frequency)

        fig, ax = plt.subplots()
        self.plot_freq_on_axis(data_frame, ax, data_frame_column_to_plot)
        ax.set_ylabel(f"|{plot_s_param.value}|")
        ax.set_xlabel("Time (s)")
        plt.title(f"|{plot_s_param.value}| Over Time at {target_frequency_GHz} GHz")

        if save_to_file:
            os.makedirs(output_folder_path, exist_ok=True)
            plt.savefig(
                os.path.join(
                    output_folder_path,
                    f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}-{target_frequency_GHz}_GHz.svg",
                ),
                format="svg",
            )
        plt.show()

    def pivot_data_frame_frequency(self, value: DataFrameCols) -> pd.DataFrame:
        return self.data_frame.pivot(
            index=DataFrameCols.TIME, columns=DataFrameCols.FREQUENCY, values=value
        )


class VNA:

    @staticmethod
    def file_label_input() -> str:
        file_label = input(
            "Input label for file (no spaces) or press enter for no label:"
        )
        while (file_label != "") and not VnaData.test_file_name(file_label):
            file_label = input("Incorrect format try again or enter to skip:")
        return file_label

    def __init__(
        self,
        calibration: VnaCalibration,
        vna_data: VnaData,
        vna_string="PicoControl2.PicoVNA_2",
    ):
        self.calibration = calibration
        self.vna_object = win32com.client.gencache.EnsureDispatch(vna_string)
        self.output_data = vna_data

    def connect(self):
        print("Connecting VNA")
        search_vna = self.vna_object.FND()
        if search_vna == 0:
            raise VNAError("Connection Failed, do you have Pico VNA Open?")
        print(f"VNA {str(search_vna)} Loaded")

    def load_cal(self):
        print("Loading Calibration")
        ans = self.vna_object.LoadCal(self.calibration.calibration_path)
        if ans != "OK":
            raise VNAError(f"Calibration Failure {ans}")
        print(f"Result {ans}")

    def get_data(
        self, s_parameter: SParam, data_format: MeasurementFormat, point=0
    ) -> str:
        """

        :param s_parameter: S Param data to be returned
        :param data_format: measurement requested
        :param point:
        :return: data string which is ',' separted in the format "freq, measurement_value_at_freq, freq, measurement_value_at_freq"
        """
        return self.vna_object.GetData(s_parameter.value, data_format.value, point)

    def split_data_string(self, data_string: str):
        """

        :param data_string: Data string which is ',' separted in the format "freq, measurement_value_at_freq, freq,
                            measurement_value_at_freq" output from self.get_data()
        :return: tuple containing list of frequencies and list of data values
        """
        data_list: list[str] = data_string.split(",")
        frequencies = data_list[::2]
        data = data_list[1::2]
        return frequencies, data

    def vna_data_string_to_df(
        self,
        elapsed_time: timedelta,
        magnitude_data_string: str,
        phase_data_string: str,
        s_parameter: SParam,
        label: str,
        id,
    ) -> pd.DataFrame:
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
        frequencies, phases = self.split_data_string(phase_data_string)
        time_float = float(f"{elapsed_time.seconds}.{elapsed_time.microseconds}")
        data_dict = {
            DataFrameCols.ID.value: id,
            DataFrameCols.TIME.value: [time_float for _ in frequencies],
            DataFrameCols.LABEL.value: [label for _ in frequencies],
            DataFrameCols.S_PARAMETER.value: [s_parameter for _ in frequencies],
            DataFrameCols.FREQUENCY.value: [int(fq) for fq in frequencies],
            DataFrameCols.MAGNITUDE.value: [float(mag) for mag in magnitudes],
            DataFrameCols.PHASE.value: [float(phase) for phase in phases],
        }
        return pd.DataFrame(data_dict)

    def generate_output_path(
        self,
        output_folder: str,
        s_params_saved: SParam,
        run_time: timedelta,
        fname="",
        label="",
    ):
        """
        Utility function to generate file name and join it ot path
        :param s_params_measure: measured s parameteres
        :param run_time:
        :param fname:
        :return:
        """
        if fname != "" and label != "":
            label_fname = ("_").join((fname, label))
        else:
            label_fname = ("").join((fname, label))

        if label == "":
            label = datetime.now().strftime(DateFormats.DATE_FOLDER.value)

        if label_fname != "":
            label_fname += "_"

        s_params = ("_").join([s_param.value for s_param in s_params_saved])
        filename = f"{label_fname}{datetime.now().strftime(DateFormats.CURRENT.value)}_{s_params}_{run_time.seconds}_secs.csv"
        return os.path.join(get_root_folder_path(), output_folder, label, filename)

    # todo what if you only want to measure one of phase or logmag?
    def add_measurement_to_data_frame(
        self, s_param: SParam, elapsed_time: timedelta, label: str, id
    ):
        """
        Gets current measurement strings (logmag and phase) for the given S param from VNA and converts it
        to a pd data frame, appending this data frame to the output data
        :param s_param: SParam to get the data
        :param elapsed_time: The elaspsed time of the current test (ie the time the data was captured, referenced to 0s)
        :return: the data frame concated on to the current output
        """
        df = self.vna_data_string_to_df(
            elapsed_time,
            self.get_data(s_param, MeasurementFormat.LOGMAG),
            self.get_data(s_param, MeasurementFormat.PHASE),
            s_param.value,
            label,
            id,
        )
        return pd.concat([self.output_data.data_frame, df])

    # add in timer logging
    @timer_func
    def measure_wrapper(self, str):
        return self.vna_object.Measure(str)

    @timer_func
    def take_measurement(
        self,
        s_params_measure: MeasureSParam,
        s_params_output: [SParam],
        elapsed_time: timedelta,
        label: str,
        id,
    ):
        """
        Takes measurement on the VNA, processes it and appends it to the output_data.data_frame
        df
        :param s_params_measure: The S params for the VNA to measure, using
        :param s_params_output:
        :param elapsed_time:
        """
        # todo what does this return? format?
        self.measure_wrapper(s_params_measure.value)

        # todo check how the measurement formats work, where is phase and logmag defined?
        for s_param in s_params_output:
            self.output_data.data_frame = self.add_measurement_to_data_frame(
                s_param, elapsed_time, label, id
            )

    def input_movement_label(self) -> str:
        label = input("Provide gesture label or leave blank for none:")
        return label

    def measure(
        self,
        run_time: timedelta,
        s_params_measure: MeasureSParam = MeasureSParam.ALL,
        s_params_output: [SParam] = None,
        file_name: str = "",
        output_dir=os.path.join("results", "data"),
        label=None,
    ) -> VnaData:

        # label = 'test'
        if label is None:
            label = self.input_movement_label()

        if s_params_output == None:
            s_params_output = [SParam.S11]

        self.output_data.csv_path = self.generate_output_path(
            output_dir, s_params_output, run_time, file_name, label
        )
        os.makedirs(os.path.dirname(self.output_data.csv_path), exist_ok=True)
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

            self.take_measurement(
                s_params_measure,
                s_params_output,
                elapsed_time,
                label,
                id=start_time.strftime(DateFormats.CURRENT.value),
            )
            measurement_number += 1
            if measurement_number % 10 == 0:
                print(
                    f"Saving df data index is {measurement_number} running for another {(finish_time - datetime.now())}"
                )
                self.output_data.data_frame.to_csv(
                    self.output_data.csv_path, index=False
                )

        self.output_data.data_frame.to_csv(self.output_data.csv_path, index=False)

        self.vna_object.CloseVNA()
        print("VNA Closed")
        return self.output_data


# class ModelProcessing:
#     class Features(Enum):
#         MEANS_ABSOLUTE_VALUE = "MAV"
#         MODIFIED_MEANS_ABS_VAL_1 = "MMAV1"
#         MODIFIED_MEANS_ABS_VAL_2 = "MMAV2"
#         MEANS_ABSOLUTE_VALUE_SLOPE = "MAVSLP"
#         ROOT_MEANS_SQUARE = "RMS"
#
#     def __init__(self, segment: pd.DataFrame):
#         self.segment = segment
#         self.feature_matrix = None
#

def pivot_data_frame_for_s_param(s_param: str, data_frame: pd.DataFrame, mag_or_phase: DataFrameCols)->pd.DataFrame:
    if (mag_or_phase is not DataFrameCols.MAGNITUDE) and (mag_or_phase is not DataFrameCols.PHASE):
        raise ValueError(f"mag_or_phase must be one of those, currently is {mag_or_phase}")
    sparam_df = data_frame[data_frame[DataFrameCols.S_PARAMETER.value] == s_param]
    new_df = sparam_df.pivot(
        index=DataFrameCols.TIME.value,
        columns=DataFrameCols.FREQUENCY.value,
        values=mag_or_phase.value,
    )
    new_df.reset_index(inplace=True)
    new_df["mag_or_phase"] = mag_or_phase.value
    new_df[DataFrameCols.S_PARAMETER.value] = s_param
    new_df[DataFrameCols.ID.value] = data_frame[DataFrameCols.ID.value]
    new_df[DataFrameCols.LABEL.value] = data_frame[DataFrameCols.LABEL.value]
    reordered_columns = [
                            DataFrameCols.ID.value,
                            DataFrameCols.LABEL.value,
                            "mag_or_phase",
                            DataFrameCols.S_PARAMETER.value,
                            DataFrameCols.TIME.value,
                            ] + list(new_df.columns[1:-4])

    new_df = new_df[
        reordered_columns
    ]
    return new_df

def make_fq_df(directory: str) -> pd.DataFrame:
    csvs = os.listdir(
        os.path.join(get_data_path(), directory)
    )
    combined_data_frame = None
    for csv_fname in csvs:
        data = VnaData(
            os.path.join(get_data_path(), directory, csv_fname)
        )
        #loop over each sparam in the file and make a pivot table then append
        for sparam in data.data_frame[DataFrameCols.S_PARAMETER.value].unique():
            pivoted_data_frame = pivot_data_frame_for_s_param(sparam, data.data_frame, DataFrameCols.MAGNITUDE)
            combined_data_frame = pd.concat((combined_data_frame, pivoted_data_frame), ignore_index=True)

            pivoted_data_frame = pivot_data_frame_for_s_param(sparam, data.data_frame, DataFrameCols.PHASE)
            combined_data_frame = pd.concat((combined_data_frame, pivoted_data_frame), ignore_index=True)

    return combined_data_frame


def combine_dfs_with_labels(directory_list, labels) -> pd.DataFrame:
    ids = [i for i in range(len(directory_list))]
    new_df = make_fq_df(directory_list.pop(0), labels.pop(0), ids.pop(0))
    for dir, label, sample_id in zip(directory_list, labels, ids):
        temp_df = make_fq_df(dir, label, sample_id)
        new_df = pd.concat((new_df, temp_df), ignore_index=True)
    return new_df


def calulate_window_size_from_seconds(
    data_frame: pd.DataFrame, length_window_seconds: float
):
    return len(data_frame[(data_frame[DataFrameCols.TIME.value] < length_window_seconds)])


def rolling_window_split(data_frame: pd.DataFrame, rolling_window_seconds: float):
    new_id_list = [i for i in range(100000)]
    new_df: pd.DataFrame = None
    movement_dict = {}

    #get each of the ids in turn
    grouped_data_id = data_frame.groupby([DataFrameCols.ID.value])

    # Iterate over the groups and store each filtered DataFrame
    for group_keys, group_data in grouped_data_id:
        window_size = rolling_window_seconds
        window_start = 0.0
        window_end = window_start + window_size
        # get the avg of a single set of time
        window_increment = group_data[(group_data[DataFrameCols.S_PARAMETER.value] == SParam.S11.value) & (group_data["mag_or_phase"] == "magnitude")][DataFrameCols.TIME.value].diff().mean()
        while window_end <= group_data[DataFrameCols.TIME.value].max():
            new_id = new_id_list.pop(0)
            windowed_df = group_data[(group_data[DataFrameCols.TIME.value] >= window_start) & (group_data[DataFrameCols.TIME.value] < window_end)]
            new_df, movement_dict = combine_windowed_df(
                                        new_df, windowed_df, new_id, movement_dict
                                    )

            window_start += window_increment
            window_end += window_increment



    # for measurement_id in ids:
    #     for mag_phase in data_frame["mag_or_phase"].unique():
    #         for s_param in data_frame[DataFrameCols.S_PARAMETER.value].unique():
    #             new_id = new_id_list.pop(0)
    #             # need to select each sparam in turn and then mag and phase in turn to make sure they are all with the same id
    #             id_frame = data_frame[(data_frame[DataFrameCols.ID.value] == measurement_id) & (data_frame[DataFrameCols.S_PARAMETER.value] == s_param) & (data_frame["mag_or_phase"] == mag_phase)]
    #             # number of indexes which map to that many seconds
    #             window_size = calulate_window_size_from_seconds(
    #                 id_frame, rolling_window_seconds
    #             )
    #             rolling_window = id_frame.rolling(window=window_size)
    #             for window_df in rolling_window:
    #                 if len(window_df) == window_size:
    #                     new_df, id_movement = combine_windowed_df(
    #                         new_df, window_df, new_id, id_movement
    #                     )
    return new_df, movement_dict


def window_split(data_frame: pd.DataFrame, window_seconds: float):
    new_id_list = [i for i in range(100000)]
    ids = data_frame["id"].unique()
    new_df: pd.DataFrame = None
    movement_dict = {}
    # get each of the ids in turn
    grouped_data_id = data_frame.groupby([DataFrameCols.ID.value])

    # Iterate over the groups and store each filtered DataFrame
    for group_keys, group_data in grouped_data_id:
        window_size = window_seconds
        window_start = 0.0
        window_end = window_start + window_size
        # get the avg of a single set of time
        window_increment = window_size
        while window_end <= group_data[DataFrameCols.TIME.value].max():
            new_id = new_id_list.pop(0)
            windowed_df = group_data[(group_data[DataFrameCols.TIME.value] >= window_start) & (
                        group_data[DataFrameCols.TIME.value] < window_end)]
            new_df, id_movement = combine_windowed_df(
                new_df, windowed_df, new_id, movement_dict
            )

            window_start += window_increment
            window_end += window_increment

    return new_df, movement_dict


def combine_windowed_df(
    new_df: pd.DataFrame, windowed_df: pd.DataFrame, new_id, movement_dict
) -> pd.DataFrame:
    windowed_df = windowed_df.reset_index(drop=True)

    windowed_df[DataFrameCols.ID.value] = new_id
    movement_dict[new_id] = windowed_df[DataFrameCols.LABEL.value][0]

    VnaData.zero_ref_time(windowed_df)
    if new_df is None:
        new_df = windowed_df
    else:
        new_df = pd.concat((new_df, windowed_df), ignore_index=True)
    return new_df, movement_dict


def extract_features_and_test(full_data_frame, feature_vector, drop_cols=[DataFrameCols.LABEL.value]):
    combined_df = full_data_frame.ffill()
    # s_params_mapping = {s.value:index+1 for index, s in enumerate(SParam)}
    # full_data_frame[DataFrameCols.S_PARAMETER.value].map({s.value: index for index, s in enumerate(SParam)})
    dropped_label = combined_df.drop(
        columns=drop_cols
    )
    extracted = extract_features(dropped_label, column_sort=DataFrameCols.TIME.value, column_id=DataFrameCols.ID.value, n_jobs=0)
    impute(extracted)
    features_filtered = select_features(extracted, feature_vector)

    X_full_train, X_full_test, y_train, y_test = train_test_split(
        extracted, feature_vector, test_size=0.4
    )

    classifier_full = DecisionTreeClassifier()
    classifier_full.fit(X_full_train, y_train)
    print(classification_report(y_test, classifier_full.predict(X_full_test)))

    X_filtered_train, X_filtered_test = (
        X_full_train[features_filtered.columns],
        X_full_test[features_filtered.columns],
    )
    classifier_filtered = DecisionTreeClassifier()
    classifier_filtered.fit(X_filtered_train, y_train)
    print(classification_report(y_test, classifier_filtered.predict(X_filtered_test)))

    print("SVM".center(80, "="))
    # Splitting the data into training and testing sets
    X_full_train, X_full_test, y_train, y_test = train_test_split(extracted, feature_vector, test_size=0.4)

    # Standardizing the feature vectors
    scaler = StandardScaler()
    X_full_train_scaled = scaler.fit_transform(X_full_train)
    X_full_test_scaled = scaler.transform(X_full_test)

    # Creating an SVM classifier
    svm_classifier = SVC()

    # Training the SVM classifier
    svm_classifier.fit(X_full_train_scaled, y_train)
    print("Full")
    # Evaluating the SVM classifier
    print(classification_report(y_test, svm_classifier.predict(X_full_test_scaled)))

    # Splitting the data into training and testing sets
    X_full_train, X_full_test, y_train, y_test = train_test_split(features_filtered, feature_vector, test_size=0.4)

    # Standardizing the feature vectors
    scaler = StandardScaler()
    X_full_train_scaled = scaler.fit_transform(X_full_train)
    X_full_test_scaled = scaler.transform(X_full_test)

    # Creating an SVM classifier
    svm_classifier_filtered = SVC()

    # Training the SVM classifier
    svm_classifier_filtered.fit(X_full_train_scaled, y_train)
    print("Filtered")
    # Evaluating the SVM classifier
    print(classification_report(y_test, svm_classifier_filtered.predict(X_full_test_scaled)))

    return {"filered_classifier": classifier_filtered, "full_classifier": classifier_full, "svm_full": svm_classifier, "svm_filtered": svm_classifier_filtered, "full_features": extracted, "filtered_features": features_filtered}

def make_columns_have_s_param_mag_phase_titles(data_frame: pd.DataFrame)->pd.DataFrame:
    freq_cols = [val for val in data_frame.columns.values if isinstance(val, int)]
    grouped_data = data_frame.groupby(["mag_or_phase", DataFrameCols.S_PARAMETER.value])
    new_combined_df = None
    for keys, df in grouped_data:
        label_to_add = ("_").join(keys)
        new_cols = [f"{label_to_add}_{col_title}" for col_title in freq_cols]
        df.rename(columns=dict(zip(freq_cols, new_cols)), inplace=True)
        df = df.drop(
            columns=[DataFrameCols.S_PARAMETER.value, "mag_or_phase"]
            )
        if new_combined_df is None:
            new_combined_df = df
        else:
            new_combined_df = pd.merge(new_combined_df, df, on=[DataFrameCols.ID.value, DataFrameCols.TIME.value, DataFrameCols.LABEL.value])
    return new_combined_df

# if __name__ == "__main__":
#     dirs = os.listdir(os.path.join(get_root_folder_path(), "data", "processed_data"))
#     label_dict = {"fist": 5, "star": 6}
#     labels = []
#     for folder in os.listdir(
#         os.path.join(get_root_folder_path(), "data", "processed_data")
#     ):
#         try:
#             labels.append(int(folder.split("_")[0][0]))
#         except ValueError as e:
#             labels.append(label_dict[folder.split("_")[1]])
#
#     print(labels)
#     combined_df = combine_dfs_with_labels(dirs, labels)
#

#
#     windowed_df, windowed_movement_dict = window_split(combined_df, 2.0)
#     windowed_movement_vector = pd.Series(windowed_movement_dict.values())
#
#     print("Rolling")
#     extract_features_and_test(rolling_df, rolling_movement_vector)
#
#     print("Windowed")
#     extract_features_and_test(windowed_df, windowed_movement_vector)
#

def filter_cols_between_fq_range(df: pd.DataFrame, lower_bound, upper_bound):
    cols = df.columns.values
    # Filter out non-integer values
    filtered_list = [x for x in cols if isinstance(x, int)]
    # Filter the list based on the provided bounds
    freq_cols = [x for x in filtered_list if lower_bound <= x <= upper_bound]
    return filter_columns(df, freq_cols)

def filter_columns(df, frequencies):
    pattern = fr'^id$|^label$|^mag_or_phase$|^s_parameter$|^time$'
    if frequencies:
        pattern += '|' + '|'.join(f'^{num}$' for num in frequencies)
    return df.filter(regex=pattern, axis=1)

def pickle_object(object_to_pickle, fname):
    with open(os.path.join(get_pickle_path(), fname), "wb") as f:
        pickle.dump(object_to_pickle, f)

def open_pickled_object(fname):
    with open(os.path.join(get_pickle_path(), fname), "wb") as f:
        unpickled = pickle.load(f)
    return unpickled

def feature_extract_test_filtered_data_frame(filtered_data_frame, save=True, fname=None):
    df_fixed = make_columns_have_s_param_mag_phase_titles(filtered_data_frame)
    classifiers = extract_features_and_test(df_fixed, windowed_movement_vector)
    if save:
        if fname is None:
            fname = f"classifier_{datetime.now().date().strftime(DateFormats.DATE_FOLDER.value)}"
        else:
            fname = f"{fname}_{datetime.now().date().strftime(DateFormats.DATE_FOLDER.value)}"
        pickle_object(classifiers, fname)
    return classifiers, fname

if __name__ == "__main__":

    # os.makedirs(get_pickle_path(), exist_ok=True)
    #
    # data_folders = os.listdir(get_data_path())
    # combined_df: pd.DataFrame = None
    # for data_folder in data_folders:
    #     combined_df_for_one_folder = make_fq_df(data_folder)
    #     combined_df = pd.concat((combined_df, combined_df_for_one_folder), ignore_index=True)
    # full_df_path = os.path.join(get_pickle_path(),"full_dfs")
    # os.makedirs(full_df_path, exist_ok=True)
    # with open(os.path.join(full_df_path, f"full_combined_df_{datetime.now().date().strftime(DateFormats.DATE_FOLDER.value)}.pkl"), "wb") as f:
    #     pickle.dump(combined_df, f)

    # rolling_df, rolling_movement = rolling_window_split(combined_df, 2.0)
    # rolling_movement_vector = pd.Series(rolling_movement.values())
    #
    # rolling_df = make_columns_have_s_param_mag_phase_titles(rolling_df)

    windowed_df, windowed_movement_dict = window_split(combined_df, 2.0)
    windowed_movement_vector = pd.Series(windowed_movement_dict.values())

    windowed_all_Sparams_magnitude = windowed_df[(windowed_df['mag_or_phase'] == "magnitude")]
    windowed_all_Sparams_magnitude_filtered = filter_cols_between_fq_range(windowed_all_Sparams_magnitude, ghz_to_hz(2), ghz_to_hz(2.6))

    print("Windowed")
    classifiers, fname = feature_extract_test_filtered_data_frame(windowed_all_Sparams_magnitude_filtered, fname="classifier_windowed_mag_all_s_2_2-6.pkl")



    # filtered_windowed_df = filter_cols_between_fq_range(windowed_df, ghz_to_hz(2.1), ghz_to_hz(2.6))
    #
    # filtered_windowed_df_cols_changed = make_columns_have_s_param_mag_phase_titles(filtered_windowed_df[(filtered_windowed_df["mag_or_phase"] == "magnitude") & (filtered_windowed_df[DataFrameCols.S_PARAMETER.value] == SParam.S11.value)])
    #
    # filtered_windowed_df = make_columns_have_s_param_mag_phase_titles(windowed_df[(windowed_df["mag_or_phase"]=="magnitude") & (windowed_df[DataFrameCols.S_PARAMETER.value]==SParam.S21.value)])

    # print("Rolling")
    # extract_features_and_test(rolling_df, rolling_movement_vector)

    # print("Windowed")
    # classifiers = extract_features_and_test(filtered_windowed_df, windowed_movement_vector)
    # with open(os.path.join(get_pickle_path(), "classifier_windowed_mag_s21.pkl"), "wb") as f:
    #     pickle.dump(classifiers, f)





    # with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "classifier_windowed_mag_s11_2.pkl"), 'rb') as f:
    #     loaded_classifier = pickle.load(f)


