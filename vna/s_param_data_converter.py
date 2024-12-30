from __future__ import annotations

import os
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from natsort import natsorted
from skrf import Network, Touchstone

from s_parameter_data import SParameterData

from VNA_data import VnaData

from VNA_enums import DataFrameCols, DateFormats


class SParamDataConverter(ABC):
    @abstractmethod
    def convert_to_s_param_data(self)->SParameterData:
        pass

class CsvSParamDataConverter(SParamDataConverter):
    def __init__(self, csv_folder_path):
        self.csv_folder_path = csv_folder_path

    def convert_to_s_param_data(self) -> SParameterData:
        """
        converts a given directory containing .csv data
        :param directory:
        :return:
        """
        csvs = os.listdir(self.csv_folder_path)
        combined_data_frame = None
        for csv_fname in csvs:
            data = VnaData(os.path.join(self.csv_folder_path, csv_fname))
            # loop over each sparam in the file and make a pivot table then append
            for sparam in data.data_frame[DataFrameCols.S_PARAMETER.value].unique():
                pivoted_data_frame = self.pivot_data_frame_for_s_param(
                    sparam, data.data_frame, DataFrameCols.MAGNITUDE
                )
                combined_data_frame = pd.concat(
                    (combined_data_frame, pivoted_data_frame), ignore_index=True
                )

                pivoted_data_frame = self.pivot_data_frame_for_s_param(
                    sparam, data.data_frame, DataFrameCols.PHASE
                )
                combined_data_frame = pd.concat(
                    (combined_data_frame, pivoted_data_frame), ignore_index=True
                )

        return SParameterData(combined_data_frame)

    def pivot_data_frame_for_s_param(
            self, s_param: str, data_frame: pd.DataFrame, mag_or_phase: DataFrameCols
    ) -> pd.DataFrame:
        """
        Takes in a data_frame in DataFrameFormats format and returns a dataframe which
        has been pivoted to have the frequency as the column title with the other info
        (ie the s param, id, label, time) in seperate columns
        :param s_param: desired sparam for filtering
        :param data_frame: dataframe to be pivoted
        :param mag_or_phase: magnitude or phase selection for pivoting
        :return: pivoted dataframe with the columns reordered
        """
        if (mag_or_phase is not DataFrameCols.MAGNITUDE) and (
                mag_or_phase is not DataFrameCols.PHASE
        ):
            raise ValueError(
                f"mag_or_phase must be one of those, currently is {mag_or_phase}"
            )
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

        new_df = new_df[reordered_columns]
        return new_df

class TouchstoneSParamDataConverter(SParamDataConverter):

    def __init__(self,  touchstone_folder_path, experiment_label):
        self.touchstone_folder_path = touchstone_folder_path
        self.experiment_label = experiment_label
        self.data_frame = None

    def convert_to_s_param_data(self) ->SParameterData:
        return self.extract_data_from_touchstone_folder(self.touchstone_folder_path, self.experiment_label)

    def extract_data_from_touchstone_folder(self, folder_path, label) -> pd.DataFrame:
        df = None
        touchstone_files = natsorted(os.listdir(folder_path))[:3]
        times = [self.get_time_recorded_from_touchstone(os.path.join(folder_path, path)) for path in touchstone_files]
        times = self.space_out_duplicates(times)
        i = 0
        for touchstone_file in touchstone_files:
            path = os.path.join(folder_path, touchstone_file)
            print(f'{i} of {len(touchstone_files)}')
            df, times = self.extract_values_from_touchstone_files_to_df(path, label, times, df)
            i += 1
        return df

    def get_time_recorded_from_touchstone(self, path):
        with open(path, 'r') as file:
            lines = file.readlines()
            if len(lines) >= 2:
                line = lines[1].strip()
                time_recorded = line.split(' ', maxsplit=1)[1]
                return datetime.fromisoformat(time_recorded)
            else:
                pass

    def s_params_as_list(self, network: Network, s_param, phase_flag=False, mag_flag=False):
        destination_port = int(s_param[1])
        origin_port = int(s_param[2])
        assert (1 <= destination_port <= 4) & (1 <= origin_port <= 4)
        if phase_flag:
            return [np.angle(matrix[destination_port - 1][origin_port - 1]) for matrix in network.s]
        elif mag_flag:
            return [abs(matrix[destination_port - 1][origin_port - 1]) for matrix in network.s]
        else:
            return [matrix[destination_port - 1][origin_port - 1] for matrix in network.s]

    def get_complex_phase_mag_lists(self, network: Network, s_param):
        complex_values = self.s_params_as_list(network, s_param)
        phase_values = self.s_params_as_list(network, s_param, phase_flag=True)
        db_values = self.s_params_as_list(network, s_param, mag_flag=True)
        return complex_values, phase_values, db_values

    def space_out_duplicates(self, datetimes):
        spaced_out = []  # This will hold the adjusted datetimes
        i = 0

        while i < len(datetimes):
            current_datetime = datetimes[i]
            spaced_out.append(current_datetime)

            # Find the next different datetime (to detect consecutive duplicates)
            duplicates = [current_datetime]
            while i + 1 < len(datetimes) and datetimes[i + 1] == current_datetime:
                duplicates.append(datetimes[i + 1])
                i += 1

            # If duplicates were found, space them out by milliseconds
            if len(duplicates) > 1:
                num_duplicates = len(duplicates)
                ms_spacing = 1000 // num_duplicates  # Divide 1000ms evenly among duplicates

                for j in range(1, num_duplicates):
                    new_datetime = current_datetime + timedelta(milliseconds=ms_spacing * j)
                    spaced_out.append(new_datetime)

            i += 1

        return spaced_out

    def zero_ref_time_column(self, df):
        df['time'] = pd.to_datetime(df['time'], format="%Y_%m_%d_%H_%M_%S.%f")
        df['time'] = df.groupby('id')['time'].transform(lambda x: x - x.min())
        df['time'] = df['time'].transform(lambda x: x.total_seconds())
        return df

    def create_empty_data_frame(self, network: Network) -> pd.DataFrame:
        frequency_array = network.f
        columns = ['id', 'label', 'mag_or_phase', 's_parameter', 'time'] + [str(int(fq)) for fq in frequency_array]
        return pd.DataFrame(columns=columns)


    def extract_values_from_touchstone_files_to_df(self,
                                                   path,
                                                   experiment_label,
                                                   times, df: pd.DataFrame = None) -> pd.DataFrame:
        touchstone_network = Network(path)
        if df is None:
            df = self.create_empty_data_frame(touchstone_network)

        fname = os.path.basename(path)
        split_fname = fname.split('_')

        experiment_id = ('_').join(split_fname[:6])

        time_touchstone_created = times.pop(0)

        gesture_label = split_fname[7]
        label = experiment_label + '_' + gesture_label

        s_params = [f'S{i}{j}' for i in range(1, 5) for j in range(1, 5)]

        for s_param in s_params:
            complex_values, phase_values, db_values = self.get_complex_phase_mag_lists(touchstone_network, s_param)
            print(f"{s_param} time {time_touchstone_created.strftime(DateFormats.MILLISECONDS.value)}")
            phase_row = [experiment_id, label, 'phase', s_param,
                         time_touchstone_created.strftime(DateFormats.MILLISECONDS.value)] + phase_values
            db_row = [experiment_id, label, 'magnitude', s_param,
                      time_touchstone_created.strftime(DateFormats.MILLISECONDS.value)] + db_values

            df.loc[len(df)] = phase_row
            df.loc[len(df)] = db_row
        return df, times


