import os
from __future__ import annotations

from pathlib import Path

import pandas as pd

from VNA_utils import get_full_df_path, open_pickled_object, mhz_to_hz
from VNA_data import VnaData
from VNA_enums import DataFrameCols, DfFilterOptions
from single_gesture_classifier import print_fq_hop


class CsvToResultsDataFrame:
    def __init__(self, csv_folder_path: Path):
        self.csv_folder_path = csv_folder_path

    def csv_directory_to_ml_data_frame(self) -> SParameterData:
        """
        converts a given directory containing .csv data
        :param directory:
        :return:
        """
        csvs = os.listdir(self.directory)
        combined_data_frame = None
        for csv_fname in csvs:
            data = VnaData(os.path.join(self.directory, csv_fname))
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

class SParameterData:

    @staticmethod
    def open_full_results_df(file_name, folder=None) -> SParameterData:
        """
        Opens a .pkl data frame within the folder provided, if folder arg is none
        then the default folder is used
        :param file_name: the file name of the target data frame
        :param folder: the folder of the data frame
        :return: data frame
        """
        if folder is None:
            folder = get_full_df_path()
        data_frame = open_pickled_object(os.path.join(folder, file_name))
        return SParameterData(data_frame)


    def __init__(self, data_frame: pd.DataFrame=None):
        self.data_frame = data_frame
        self.data_frame.columns = list(self.data_frame.columns[:5]) + [int(x) for x in self.data_frame.columns[5:]]
        self.data_frame_split_by_id = None


    def split_data_frame_into_id_chunks(
            self, ids_per_split: int
    ) -> [pd.DataFrame]:

        # Get the unique IDs
        unique_ids = self.data_frame[DataFrameCols.ID.value].unique()

        # Initialize a list to store the smaller DataFrames
        split_dfs_by_id = []

        # Split into chunks of 3 IDs each
        for i in range(0, len(unique_ids), ids_per_split):
            # Get the current chunk of 3 IDs
            chunk_ids = unique_ids[i: i + ids_per_split]

            # Filter the original DataFrame for those IDs
            smaller_df = self.data_frame[self.data_frame[DataFrameCols.ID.value].isin(chunk_ids)]

            # Append the resulting DataFrame to the list
            split_dfs_by_id.append(smaller_df)

        self.data_frame_split_by_id = split_dfs_by_id
        return split_dfs_by_id

class Classifier:
    def __init__(self, full_results: SParameterData):
        self.full_results = full_results
        self.filtered_results_dict = None

    def create_test_dict(self,
                         sparam_sets: list[list[str]],
                         filter_type: DfFilterOptions = DfFilterOptions.BOTH) -> dict:
        """
        This function creates the test dict for the classifier, allowing filtering by specific S-parameter sets
        and by magnitude, phase, or both.

        :param sparam_sets: A list of lists containing S-parameter strings (e.g., [['S11', 'S12'], ['S21']]).
        :param filter_type: Filter by 'magnitude', 'phase', or 'both'. Defaults to 'both'.
        :return: A dictionary with filtered dataframes.
        """
        results_data_frame = self.full_results.data_frame

        # Initialize the dictionary to store filtered dataframes
        self.filtered_results_dict = {}

        # Check the filter type and set which columns to filter
        if filter_type in [DfFilterOptions.BOTH, DfFilterOptions.MAGNITUDE]:
            all_Sparams_magnitude = results_data_frame[results_data_frame["mag_or_phase"] == "magnitude"]
        if filter_type in [DfFilterOptions.BOTH, DfFilterOptions.MAGNITUDE]:
            all_Sparams_phase = results_data_frame[results_data_frame["mag_or_phase"] == "phase"]

        # Iterate over each sparameter set provided in sparam_sets
        for i, sparam_set in enumerate(sparam_sets):
            set_name = f"{('_').join(sparam_set)}"

            # Filter for magnitude if specified or 'both'
            if filter_type in [DfFilterOptions.BOTH, DfFilterOptions.MAGNITUDE]:
                self.filtered_results_dict[f"{set_name}_magnitude"] = all_Sparams_magnitude[
                    all_Sparams_magnitude[DataFrameCols.S_PARAMETER.value].isin(sparam_set)
                ]

            # Filter for phase if specified or 'both'
            if filter_type in [DfFilterOptions.BOTH, DfFilterOptions.MAGNITUDE]:
                self.filtered_results_dict[f"{set_name}_phase"] = all_Sparams_phase[
                    all_Sparams_phase[DataFrameCols.S_PARAMETER.value].isin(sparam_set)
                ]

            if filter_type in [DfFilterOptions.BOTH]:
                self.filtered_results_dict[f"{set_name}_both"] = results_data_frame[
                    results_data_frame[DataFrameCols.S_PARAMETER.value].isin(sparam_set)
                ]

        return self.filtered_results_dict

