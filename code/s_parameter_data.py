import os
from __future__ import annotations

import numpy as np

import uuid
from argparse import ArgumentError

import pandas as pd

from VNA_utils import get_full_df_path, open_pickled_object
from VNA_enums import DataFrameCols, DfFilterOptions
from movement_vector import MovementVector
from frequency import Frequency


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

    def __init__(self, label: str, data_frame: pd.DataFrame):
        self.data_frame: pd.DataFrame = data_frame
        self.data_frame.columns = list(self.data_frame.columns[:5]) + [
            int(x) for x in self.data_frame.columns[5:]
        ]
        self.minimum_frequency = Frequency(
            min(self.get_frequency_column_headings_list())
        )
        self.maximum_frequency = Frequency(
            max(self.get_frequency_column_headings_list())
        )

        self.label: str = label
        self.data_frame_split_by_id: [SParameterData] = None
        self.id: uuid.UUID = uuid.uuid4()
        self.movement_vector: MovementVector = self.create_movement_vector()

    def __str__(self):
        return f"Data containing: {self.label}, UUID: {self.id}"

    def __repr__(self):
        return f"SParameterData({self.label}, {self.data_frame}) UUID: {self.id}"

    def get_minimum_frequency(self) -> Frequency:
        return self.minimum_frequency

    def get_maximum_frequency(self) -> Frequency:
        return self.maximum_frequency

    def get_full_data_frame(self):
        return self.data_frame

    def get_magnitude_data_frame(self) -> pd.DataFrame:
        return self.data_frame[self.data_frame["mag_or_phase"] == "magnitude"]

    def get_phase_data_frame(self) -> pd.DataFrame:
        return self.data_frame[self.data_frame["mag_or_phase"] == "phase"]

    def get_frequency_column_headings_list(self) -> [Frequency]:
        return [Frequency(x) for x in self._get_frequency_columns()]

    def _get_frequency_columns(self) -> [int]:
        return [
            x
            for x in self.data_frame.columns.values
            if isinstance(x, int) or isinstance(x, np.int64)
        ]

    def split_data_frame_into_n_id_chunks(self, ids_per_split: int) -> [SParameterData]:
        """
        Splits the full data frame into a list of SParameterData objects containing at most ids_per_split
        objects, this is for feature extraction
        Args:
            ids_per_split: the max number of ids per split

        Returns:
            List of SParameterData objects split, also adds list to self.data_frame_split_by_id

        """
        if self.data_frame is None:
            raise ArgumentError("Data frame can't be None")

        # Get the unique IDs
        unique_ids = self.data_frame[DataFrameCols.ID.value].unique()

        # Initialize a list to store the smaller DataFrames
        split_dfs_by_id = []

        # Split into chunks of 3 IDs each
        for i in range(0, len(unique_ids), ids_per_split):
            # Get the current chunk of 3 IDs
            chunk_ids = unique_ids[i : i + ids_per_split]

            # Filter the original DataFrame for those IDs
            smaller_df = self.data_frame[
                self.data_frame[DataFrameCols.ID.value].isin(chunk_ids)
            ]
            label = f"{self.label} split {i}/{len(unique_ids)//ids_per_split}"

            data_object = SParameterData(label, smaller_df)
            # Append the resulting DataFrame to the list
            split_dfs_by_id.append(data_object)

        self.data_frame_split_by_id = split_dfs_by_id
        return split_dfs_by_id

    def get_data_frame_between_frequency(
        self, low_frequency: Frequency, high_frequency: Frequency
    ) -> SParameterData:
        """
        Filter the data frame so only the fq window of interest is selected and that the
        frequencies are in the range of the data frame
        :param lower_bound: Lower frequency bound
        :param upper_bound: Higher frequency bound
        :return: SParameterData object containing just the frequencies
        """
        freq_cols: [Frequency] = [
            Frequency(x)
            for x in self._get_frequency_columns()
            if low_frequency.get_freq_hz() <= x <= high_frequency.get_freq_hz()
        ]
        if not freq_cols:
            raise ValueError(
                f"The frequencies {low_frequency} and {high_frequency} are not in the range of the data"
            )
        return self._filter_columns_between_frequencies(freq_cols)

    def _filter_columns_between_frequencies(
        self, *, filter_frequencies: [Frequency]
    ) -> SParameterData:
        """
        To account for the mix of titles, this regex is used. It will match
        any of the column headings and then also the list of column frequencies
        that you pass in, returning a SParameterData object.

        This is a helper method for get_data_frame_between_frequency()
        :param filter_frequencies: List of Frequency objects which you want to filter by
        :return: SParameterData object with the filtered frequencies
        """
        pattern = rf"^id$|^label$|^mag_or_phase$|^s_parameter$|^time$"
        if filter_frequencies:
            pattern += "|" + "|".join(
                f"^{frequency.get_freq_hz()}$" for frequency in filter_frequencies
            )
        filtered_df = self.data_frame.filter(regex=pattern, axis=1)
        new_label = (
            self.label
            + f"filtered between {filter_frequencies[0].get_freq_hz()}Hz and {filter_frequencies[-1].get_freq_hz()}Hz"
        )
        return SParameterData(filtered_df, label=new_label)

    def create_movement_vector(self) -> MovementVector:
        """
        Creates a movement vector which maps each unique ID to its associated gesture for classification
        Returns:
            Movement vector object

        """
        if self.data_frame is None:
            raise ArgumentError("Data frame can't be None")

        return MovementVector.create_movement_vector_for_single_data_frame(
            df=self.data_frame
        )


class Classifier:
    def __init__(self, full_results: SParameterData):
        self.full_results = full_results
        self.filtered_results_dict = None

    def create_test_dict(
        self,
        sparam_sets: list[list[str]],
        filter_type: DfFilterOptions = DfFilterOptions.BOTH,
    ) -> dict:
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
            all_Sparams_magnitude = results_data_frame[
                results_data_frame["mag_or_phase"] == "magnitude"
            ]
        if filter_type in [DfFilterOptions.BOTH, DfFilterOptions.MAGNITUDE]:
            all_Sparams_phase = results_data_frame[
                results_data_frame["mag_or_phase"] == "phase"
            ]

        # Iterate over each sparameter set provided in sparam_sets
        for i, sparam_set in enumerate(sparam_sets):
            set_name = f"{('_').join(sparam_set)}"

            # Filter for magnitude if specified or 'both'
            if filter_type in [DfFilterOptions.BOTH, DfFilterOptions.MAGNITUDE]:
                self.filtered_results_dict[f"{set_name}_magnitude"] = (
                    all_Sparams_magnitude[
                        all_Sparams_magnitude[DataFrameCols.S_PARAMETER.value].isin(
                            sparam_set
                        )
                    ]
                )

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
