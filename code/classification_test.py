from typing import Union, List, Dict

import pandas as pd

from VNA_utils import mhz_to_hz
from VNA_enums import FourPortSParams, DataFrameCols
from code.VNA_enums import DfFilterOptions
from s_param_data_converter import SParamDataConverter
from s_parameter_data import SParameterData
from movement_vector import MovementVector



class SParameterCombinationsList:

    def __init__(self, param_list:List[List[Union[FourPortSParams, str]]]):
        self.list = self.list = [
            param if isinstance(param, FourPortSParams)
            else FourPortSParams[param.upper()]
            for s_param_list in param_list
            for param in s_param_list
        ]

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index < len(self.list):
            result = self.list[self._index]
            self._index += 1
            return result
        else:
            raise StopIteration

class ClassificationExperimentParameters:

    def __init__(self,
                 s_param_data: SParameterData,
                 s_param_combinations_list:SParameterCombinationsList,
                 s_param_measurement_options:DfFilterOptions,
                 freq_hop:int):
        self.s_param_data = s_param_data
        self.s_param_combinations_list = s_param_combinations_list
        self.s_param_measurement_options = s_param_measurement_options
        self.freq_hop = freq_hop

        self.movement_vector = MovementVector()
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
                filtered_df = all_Sparams_magnitude[
                    all_Sparams_magnitude[DataFrameCols.S_PARAMETER.value].isin(s_param_set)
                ]
                label = f"{set_name}_magnitude"
                filtered_df_dict[label] = SParameterData(label=label, data_frame=filtered_df)

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


class ClassificationExperiment:

    def __init__(self, experiment_parameters:ClassificationExperimentParameters):
        self.experiment_parameters = experiment_parameters
        self.experiment_results = ClassificationExperimentResults()

    def run_test(self):
        pass

    #todo what is data_frame? -> I think its s_parameter_data?
    def test_data_frame_classifier_frequency_window_with_report(
            self,
            label: str,
            frequency_hop: int = mhz_to_hz(100)
            ) -> pd.DataFrame:

        self.movement_vector.create_movement_vector_for_single_data_frame(data_frame)
        # as df format is | labels | fq1 | fq2 ......
        # need to get just the fqs which are listed
        freq_list = get_frequency_column_headings_list(data_frame)

        min_frequency, max_frequency = min(freq_list), max(freq_list)
        low_frequency, high_frequency = min_frequency, min_frequency + frequency_hop

        f1_scores = {}

        while high_frequency <= max_frequency:
            print_fq_hop(high_frequency, label, low_frequency)

            #
            data_frame_fq_range_filtered = filter_cols_between_fq_range(
                data_frame, low_frequency, high_frequency
            )
            fq_label = f"{label}_{hz_to_ghz(low_frequency)}_{hz_to_ghz(high_frequency)}"
            result, fname = feature_extract_test_filtered_data_frame(
                data_frame_fq_range_filtered, movement_vector, fname=fq_label
            )
            f1_scores[fq_label] = extract_report_dictionary_from_test_results(result)
            low_frequency += frequency_hop
            high_frequency += frequency_hop
        return pd.DataFrame.from_dict(
            f1_scores, orient="index", columns=[x for x in result.keys() if "report" in x]
        )

class ClassificationExperimentResults:

    def __init__(self, results_df:pd.DataFrame=None):
        self.results_df = results_df