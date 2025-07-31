from __future__ import annotations

import re
import uuid
from argparse import ArgumentError
from pathlib import Path
from typing import Self, Optional

import numpy as np
import pandas as pd
from pandas.core.groupby import DataFrameGroupBy

from vna.VNA_enums import DataFrameCols, DfFilterOptions, DfAxis
from vna.VNA_utils import load_pickled_object_as_type
from vna.frequency import Frequency
from vna.movement_vector import MovementVector, MovementVectorPandas


class SParameterData:
    @classmethod
    def open_full_results_df(cls, label: str, path: Path) -> Self:
        """
        Opens a .pkl data frame from pathlib.Path provided
        :param path: the pathlib.Path object representing a path to a dataframe
        :return: SParameterData or subclass object
        """
        data_frame: pd.DataFrame = load_pickled_object_as_type(path, pd.DataFrame)
        return cls(label, data_frame)

    def __init__(self, label: str, data_frame: pd.DataFrame):
        self._data_frame: pd.DataFrame = data_frame
        self._label: str = label
        self.id: uuid.UUID = uuid.uuid4()

        # need to just make sure df columns are int not string type because of old impls
        self._data_frame.columns = self.convert_frequency_columns_to_int_type()

        self._minimum_frequency: Frequency = self.get_minimum_frequency_from_df()
        self._maximum_frequency: Frequency = self.get_maximum_frequency_from_df()

        self.data_frame_split_by_id: Optional[list[SParameterData]] = None

        self.movement_vector: MovementVector = self.create_movement_vector()

    @property
    def label(self) -> str:
        return self._label

    @property
    def data_frame(self) -> pd.DataFrame:
        return self._data_frame

    @property
    def minimum_frequency(self) -> Frequency:
        return self._minimum_frequency

    @property
    def maximum_frequency(self) -> Frequency:
        return self._maximum_frequency

    def __str__(self) -> str:
        return f"Data containing: {self.label}, UUID: {self.id}"

    def __repr__(self) -> str:
        return f"SParameterData({self.label}, {self.data_frame}) UUID: {self.id}"

    def _make_new_instance(self, label: str, data_frame: pd.DataFrame) -> Self:
        """
        Generates a new instance of SParameterData.
        Args:
            label: label for the dataframe
            data_frame: the dataframe which contains the sparameter data

        Returns:
            A new SParameterData instance.
        """
        return self.__class__(label, data_frame)

    def get_minimum_frequency_from_df(self) -> Frequency:
        return Frequency(min(self.get_frequency_column_headings_list()))

    def get_maximum_frequency_from_df(self) -> Frequency:
        return Frequency(max(self.get_frequency_column_headings_list()))

    def get_data_frame_between_frequency(
        self, low_frequency: Frequency, high_frequency: Frequency
    ) -> Self:
        """
        Filter the data frame so only the fq window of interest is selected and that the
        frequencies are in the range of the data frame

        Args:
            high_frequency: Frequency object representing the highest frequency
            low_frequency: Frequency object representing the lowest frequency
        Return:
            SParameterData object containing just the data between these frequencies
        """
        freq_cols: [Frequency] = [
            Frequency(x)
            for x in self.get_frequency_columns()
            if low_frequency.get_freq_hz() <= x <= high_frequency.get_freq_hz()
        ]
        if not freq_cols:
            raise ValueError(
                f"The frequencies {low_frequency} and {high_frequency} are not in the range of the data"
            )
        return self.filter_columns_between_frequencies(filter_frequencies=freq_cols)

    def generate_new_label(self, new_label: str) -> str:
        if new_label is None or (new_label.casefold() == self.label.casefold()):
            new_label = self.label
        if self.label in new_label:
            pass
        else:
            new_label = f"{self.label} {new_label}"
        return new_label

    def get_magnitude_data_frame(self) -> pd.DataFrame:
        return self.data_frame[self.data_frame["mag_or_phase"] == "magnitude"]

    def get_phase_data_frame(self) -> pd.DataFrame:
        return self.data_frame[self.data_frame["mag_or_phase"] == "phase"]

    def get_frequency_columns(self) -> list[int]:
        return [
            x
            for x in self.data_frame.columns.values
            if isinstance(x, int) or isinstance(x, np.int64)
        ]

    def get_frequency_column_headings_list(self) -> list[Frequency]:
        return [Frequency(x) for x in self.get_frequency_columns()]

    def split_data_frame_into_n_id_chunks(self, ids_per_split: int) -> list[Self]:
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

        # Split into chunks of n IDs each
        for i in range(0, len(unique_ids), ids_per_split):
            # Get the current chunk of n IDs
            chunk_ids = unique_ids[i : i + ids_per_split]

            # Filter the original DataFrame for those IDs
            smaller_df = self.data_frame[
                self.data_frame[DataFrameCols.ID.value].isin(chunk_ids)
            ]
            label = f"{self.label} split {i}/{len(unique_ids) // ids_per_split}"

            data_object = self._make_new_instance(label, smaller_df)
            # Append the resulting DataFrame to the list
            split_dfs_by_id.append(data_object)

        self.data_frame_split_by_id = split_dfs_by_id
        return split_dfs_by_id

    def filter_columns_between_frequencies(
        self, filter_frequencies: list[Frequency]
    ) -> Self:
        """
        This is a helper method for get_data_frame_between_frequency()
        :param filter_frequencies: List of Frequency objects which you want to filter by
        :return: SParameterData object with the filtered frequencies
        """
        string_cols_regex: re.Pattern = self.get_string_column_titles_regex()
        freq_cols_regex: re.Pattern = self.get_freq_cols_regex_from_list(
            filter_frequencies
        )

        string_and_freq_cols_regex = re.compile(
            string_cols_regex.pattern + "|" + freq_cols_regex.pattern
        )
        label_to_add = f" filtered between {filter_frequencies[0].get_freq_hz()}Hz and {filter_frequencies[-1].get_freq_hz()}Hz"

        return self.filter_columns_from_regex(string_and_freq_cols_regex, label_to_add)

    def create_movement_vector(self) -> MovementVector:
        """
        Creates a movement vector which maps each unique ID to its associated gesture for classification
        Returns:
            Movement vector object

        """
        return MovementVectorPandas.create_movement_vector_for_single_data_frame(self)

    def get_filtered_df_by_s_param_and_frequency(
        self,
        filter_options: DfFilterOptions,
        low_frequency: Frequency,
        high_frequency: Frequency,
    ) -> Self:
        if filter_options == DfFilterOptions.PHASE:
            output_df = self.get_phase_data_frame()
        elif filter_options == DfFilterOptions.MAGNITUDE:
            output_df = self.get_magnitude_data_frame()
        else:
            output_df = self.data_frame

        # Create a new SParameterData object to chain the next operation
        temp_data_object = self._make_new_instance(
            f"{self.label}_{filter_options.value}", output_df
        )

        # Now call get_data_frame_between_frequency on the new object
        filtered_by_freq_object = temp_data_object.get_data_frame_between_frequency(
            low_frequency, high_frequency
        )

        # Create the final object with a descriptive label
        final_label = f"{self.label}_{filter_options.value}_{low_frequency.get_freq_mhz()}-{high_frequency.get_freq_mhz()}MHz"
        return self._make_new_instance(
            final_label, filtered_by_freq_object.data_frame
        )

    def convert_frequency_columns_to_int_type(self) -> list:
        # Assuming first 5 columns are metadata, which seems to be the case from the original code.
        # This might need to be more robust.
        return list(self.data_frame.columns[:5]) + [
            int(x) for x in self.data_frame.columns[5:]
        ]

    def group_by_id(self) -> DataFrameGroupBy:
        return self.data_frame.groupby(DataFrameCols.ID.value)

    def group_by_passthrough(self, *args, **kwargs) -> DataFrameGroupBy:
        """
        This method is for a passthrough to the pandas groupby function.
        """
        return self.data_frame.groupby(*args, **kwargs)

    def create_column(self, column_name: str, value):
        # This is not ideal, as it modifies the dataframe in place.
        # A better implementation would return a new SParameterData object.
        self.data_frame[column_name] = value

    def zero_ref_times(self):
        # This is not ideal, as it modifies the dataframe in place.
        # A better implementation would return a new SParameterData object.
        self.data_frame[
            DataFrameCols.TIME.value
        ] = self.data_frame.groupby(DataFrameCols.ID.value)[
            DataFrameCols.TIME.value
        ].transform(
            lambda x: x - x.min()
        )

    def get_string_column_titles_regex(self) -> re.Pattern:
        return re.compile(rf"^id$|^label$|^mag_or_phase$|^s_parameter$|^time$")

    def get_freq_cols_regex_from_list(self, freq_list: list[Frequency]) -> re.Pattern:
        return re.compile(
            "|".join(f"^{frequency.get_freq_hz()}$" for frequency in freq_list)
        )

    def filter_columns_from_regex(
        self, regex: re.Pattern, new_label: str = None
    ) -> Self:
        data_label = self.generate_new_label(new_label)
        filtered_df = self.data_frame.filter(
            regex=regex.pattern, axis=DfAxis.COLUMN.value
        )
        return self._make_new_instance(data_label, filtered_df)

    def make_columns_have_s_param_mag_phase_titles(self) -> Self:
        """
        This function fixes something to do with the feature extraction but I genuinely cannot
        remember what it is to be honest, use it before you call feature extraction
        Returns:

        """
        data_frame = self.data_frame.copy()  # Avoid modifying the original dataframe
        freq_cols = self.get_frequency_columns()
        grouped_data = data_frame.groupby(
            ["mag_or_phase", DataFrameCols.S_PARAMETER.value]
        )
        new_combined_df: pd.DataFrame = None
        for keys, df in grouped_data:
            label_to_add = ("_").join(keys)
            new_cols = [f"{label_to_add}_{col_title}" for col_title in freq_cols]
            df = df.rename(columns=dict(zip(freq_cols, new_cols)))
            df = df.drop(columns=[DataFrameCols.S_PARAMETER.value, "mag_or_phase"])
            if new_combined_df is None:
                new_combined_df = df
            else:
                new_combined_df = pd.merge(
                    new_combined_df,
                    df,
                    on=[
                        DataFrameCols.ID.value,
                        DataFrameCols.TIME.value,
                        DataFrameCols.LABEL.value,
                    ],
                )
        return self._make_new_instance(
            label=f"{self.label} for feature extraction", data_frame=new_combined_df
        )


class NotClassifier:
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

        all_Sparams_magnitude = None
        all_Sparams_phase = None

        # Check the filter type and set which columns to filter
        if filter_type in [DfFilterOptions.BOTH, DfFilterOptions.MAGNITUDE]:
            all_Sparams_magnitude = results_data_frame[
                results_data_frame["mag_or_phase"] == "magnitude"
            ]
        if filter_type in [DfFilterOptions.BOTH, DfFilterOptions.PHASE]:
            all_Sparams_phase = results_data_frame[
                results_data_frame["mag_or_phase"] == "phase"
            ]

        # Iterate over each sparameter set provided in sparam_sets
        for i, sparam_set in enumerate(sparam_sets):
            set_name = f"{('_').join(sparam_set)}"

            # Filter for magnitude if specified or 'both'
            if filter_type in [
                DfFilterOptions.BOTH,
                DfFilterOptions.MAGNITUDE,
            ] and all_Sparams_magnitude is not None:
                self.filtered_results_dict[f"{set_name}_magnitude"] = (
                    all_Sparams_magnitude[
                        all_Sparams_magnitude[DataFrameCols.S_PARAMETER.value].isin(
                            sparam_set
                        )
                    ]
                )

            # Filter for phase if specified or 'both'
            if filter_type in [
                DfFilterOptions.BOTH,
                DfFilterOptions.PHASE,
            ] and all_Sparams_phase is not None:
                self.filtered_results_dict[f"{set_name}_phase"] = all_Sparams_phase[
                    all_Sparams_phase[DataFrameCols.S_PARAMETER.value].isin(sparam_set)
                ]

            if filter_type in [DfFilterOptions.BOTH]:
                self.filtered_results_dict[f"{set_name}_both"] = results_data_frame[
                    results_data_frame[DataFrameCols.S_PARAMETER.value].isin(sparam_set)
                ]

        return self.filtered_results_dict

