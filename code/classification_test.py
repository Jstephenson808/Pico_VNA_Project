import pandas as pd

from VNA_utils import mhz_to_hz
from s_parameter_data import SParameterData


class ClassificationTest:


    def __init__(self, freq_hop, s_param_data: SParameterData):
        self.results = s_param_data
        self.freq_hop = freq_hop


    def test_data_frame_classifier_frequency_window_with_report(
            self,
            label: str,
            frequency_hop: int = mhz_to_hz(100)
            ) -> pd.DataFrame:

        self.movement_vector = self.create_movement_vector_for_single_data_frame(data_frame)
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