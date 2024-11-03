from typing import Dict

from s_parameter_data import SParameterData

from code.s_parameter_combination_list import SParameterCombinationsList

from VNA_enums import DfFilterOptions, DataFrameCols

from movement_vector import MovementVector


class ClassificationExperimentParameters:

    def __init__(self,
                 s_param_data: SParameterData,
                 s_param_combinations_list:SParameterCombinationsList,
                 s_param_measurement_options:DfFilterOptions,
                 freq_hop: int
                 ):
        self.s_param_data = s_param_data
        self.s_param_combinations_list = s_param_combinations_list
        self.s_param_measurement_options = s_param_measurement_options
        self.freq_hop = freq_hop

        self.movement_vector = MovementVector()
        self.movement_vector.create_movement_vector_for_single_data_frame(
            self.s_param_data.get_full_data_frame())
        self.test_data_frames_dict = None

    def create_test_dict(self) -> Dict[str, SParameterData]:
        """
        This function creates the test dict for the classifier, allowing filtering by specific S-parameter sets
        and by magnitude, phase, or both.

        :param combined_df: The combined dataframe containing data.
        :param sparam_sets: A list of lists containing S-parameter strings (e.g., [['S11', 'S12'], ['S21']]).
        :param filter_type: Filter by 'magnitude', 'phase', or 'both'. Defaults to 'both'.
        :return: A dictionary with filtered dataframes.
        """

        # Initialize the dictionary to store filtered dataframes
        filtered_df_dict = {}

        # Check the filter type and set which columns to filter
        if self.s_param_measurement_options in (DfFilterOptions.BOTH, DfFilterOptions.MAGNITUDE):
            all_Sparams_magnitude = self.s_param_data.get_magnitude_data_frame()
        if self.s_param_measurement_options in (DfFilterOptions.BOTH, DfFilterOptions.MAGNITUDE):
            all_Sparams_phase = self.s_param_data.get_phase_data_frame()

        # Iterate over each sparameter set provided in sparam_sets
        for i, s_param_set in enumerate(self.s_param_combinations_list):
            set_name = f"{'_'.join([s_param.value for s_param in s_param_set])}"

            # Filter for magnitude if specified or 'both'
            if self.s_param_measurement_options in (DfFilterOptions.BOTH, DfFilterOptions.MAGNITUDE):
                self.filter_df_for_s_param_set(all_Sparams_magnitude, filtered_df_dict, s_param_set, set_name)

            # Filter for phase if specified or 'both'
            if self.s_param_measurement_options in (DfFilterOptions.BOTH, DfFilterOptions.MAGNITUDE):
                filtered_df = all_Sparams_phase[
                    all_Sparams_phase[DataFrameCols.S_PARAMETER.value].isin(s_param_set)
                ]
                label = f"{set_name}_phase"
                filtered_df_dict[label] = SParameterData(label=label, data_frame=filtered_df)

            if self.s_param_measurement_options in DfFilterOptions.BOTH:
                filtered_df = self.s_param_data.data_frame[self.s_param_data.data_frame[DataFrameCols.S_PARAMETER.value].isin(s_param_set)]
                label = f"{set_name}_both"
                filtered_df_dict[label] = SParameterData(label=label, data_frame=filtered_df)
        self.test_data_frames_dict = filtered_df_dict
        return filtered_df_dict

    def filter_df_for_s_param_set(self, all_Sparams_magnitude, filtered_df_dict, s_param_set, set_name):
        filtered_df = all_Sparams_magnitude[
            all_Sparams_magnitude[DataFrameCols.S_PARAMETER.value].isin(s_param_set)
        ]
        label = f"{set_name}_magnitude"
        filtered_df_dict[label] = SParameterData(label=label, data_frame=filtered_df)
