import ast
import os
import re
from datetime import datetime, timedelta

import numpy as np
import pandas
import pandas as pd

import matplotlib

from VNA_enums import DataFrameCols

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from VNA_enums import DataFrameCols, DateFormats, SParam, MeasurementFormat
from VNA_exceptions import NotValidCSVException, NotValidSParamException
from VNA_utils import get_root_folder_path, hz_to_ghz, ghz_to_hz


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
        correct format which is one that contains only words, "-" characters and "." characters
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
        string, date_format: str = DateFormats.ORIGINAL.value
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
            # pattern matches date format in file name
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
        """
        called if there is a path to a .csv file passed to initalize the df and the datetime of the object
        :return:
        """
        self.data_frame, self.date_time = VnaData.read_df_from_csv(self.csv_path)


    def get_first_index_of_time(self, target_time, target_magnitude=None):
        """
        This function will return the index which is after the target time, it can additionally be passed
        a target magnitude and this will return the first index which is above the target magnitude
        but also after the given time
        :param target_time:
        :param target_magnitude:
        :return:
        """
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

    def plot_frequencies(
        self,
        freq_list: [int],
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
        plots a list of frequencies on a single axis, will save to file if save_to_file is True
        :param freq_list:
        :param output_folder_path:
        :param plot_s_param:
        :param data_frame_column_to_plot:
        :param save_to_file:
        :return:
        """

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


    def add_measurement_to_data_frame(
            self, s_param: SParam,  magnitude_data_string: str,
            phase_data_string: str, elapsed_time: timedelta, label: str, id
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
            magnitude_data_string,
            phase_data_string,
            s_param.value,
            label,
            id,
        )
        self.data_frame = pd.concat([self.data_frame, df])

if __name__ == '__main__':
    pass
    # targets = []
    # data = VnaData(r'D:\James\documents\OneDrive - University of Glasgow\Glasgow\Year 2\Web App Dev 2\Workspace\picosdk-picovna-python-examples\results\data\wfa-140KHz-1001pts-10Mto4G_2\single_flex-antenna-watch-140KHz-1001pts-10Mto4G_2_2024_04_12_16_55_49_S11_S21_S12_S22_2_secs.csv')
    # data.single_freq_plotter(ghz_to_hz(0.4), plot_s_param=SParam.S11, data_frame_column_to_plot=DataFrameCols.PHASE)
    # combined_df = combine_data_frames_from_csv_folder(r'D:\James\documents\OneDrive - University of Glasgow\Glasgow\Year 2\Web App Dev 2\Workspace\picosdk-picovna-python-examples\results\data\flex')
    # combined_df['label'] = combined_df['label'].map(lambda x: x.split('_')[2])




