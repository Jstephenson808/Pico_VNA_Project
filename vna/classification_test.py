import pandas as pd

from classification_experiment_parameters import ClassificationExperimentParameters
from s_parameter_data import SParameterData
from frequency import Frequency

from feature_extractor import FeatureExtractor


class ClassificationExperimentLowerLevel:
    def __init__(
        self,
        s_param_data_under_test: SParameterData,
        test_label: str,
        frequency_hop: Frequency,
        feature_extractor: FeatureExtractor,
    ):
        self.s_param_data_under_test: SParameterData = s_param_data_under_test
        self.test_label: str = test_label
        self.frequency_hop: Frequency = frequency_hop
        self.feature_extractor: FeatureExtractor = feature_extractor

        # todo this needs to be in a lower class for experiment

    def test_data_frame_classifier_frequency_window_with_report(self) -> pd.DataFrame:
        """
        This is a copy over from the previous implementation
        Handles all testing and classification, feels like this should be a few more methods really
        Returns:

        """

        # unpack parameters for some clarity
        s_param_data: SParameterData = self.s_param_data_under_test

        low_frequency: Frequency = s_param_data.get_minimum_frequency()

        # jump by frequency hop each time
        high_frequency: Frequency = (
            s_param_data.get_minimum_frequency().get_freq_hz() + self.frequency_hop
        )

        max_frequency: Frequency = s_param_data.get_maximum_frequency()
        f1_scores = {}

        while high_frequency <= max_frequency:
            self.print_fq_hop(high_frequency, self.test_label, low_frequency)

            try:
                data_frame_fq_range_filtered = (
                    s_param_data.get_data_frame_between_frequency(
                        low_frequency, high_frequency
                    )
                )
            except ValueError as e:
                print(e)
                continue

            # This fixes the column titles for feature extraction purposes
            data_frame_fq_range_filtered.make_columns_have_s_param_mag_phase_titles()

            # label for this fq band and test
            fq_label = f"{self.test_label}_{low_frequency.get_freq_ghz()}GHz_{high_frequency.get_freq_ghz()}GHz"

            # extract features from time series
            fq_range_filtered_df_extracted_features = (
                self.feature_extractor.extract_features(data_frame_fq_range_filtered)
            )

            # now need to do the test

            result, fname = feature_extract_test_filtered_data_frame(
                data_frame_fq_range_filtered, movement_vector, fname=fq_label
            )
            f1_scores[fq_label] = extract_report_dictionary_from_test_results(result)
            low_frequency += frequency_hop
            high_frequency += frequency_hop
        return pd.DataFrame.from_dict(
            f1_scores,
            orient="index",
            columns=[x for x in result.keys() if "report" in x],
        )

    def print_fq_hop(
        self, high_frequency: Frequency, label: str, low_frequency: Frequency
    ):
        print(
            f"{label}\n\r{low_frequency.get_freq_ghz()}GHz->{high_frequency.get_freq_ghz()}GHz"
        )


# design here is that each classification test has it's own one of these objects,
# will extract features etc for each
class ClassificationExperimentResults:
    def __init__(self, data_frame):
        self.data_frame = data_frame


class ClassificationExperiment:

    def __init__(
        self,
        experiment_parameters: ClassificationExperimentParameters,
        feature_extractor: FeatureExtractor = None,
    ):
        self.experiment_parameters = experiment_parameters
        self.experiment_results = ClassificationExperimentResults()
        self.feature_extractor = feature_extractor

    def run_experiment(self):
        # this is per freq hop -> I think this should be how it works,
        # higher class handles the freq windowing etc
        if self.feature_extractor:
            self.feature_extractor.extract_features()

    def test_classifier_from_df_dict(self) -> ClassificationExperimentResults:
        """
        This returns a report and save classifier to pkl path
        """
        full_results_df = None
        for (
            label,
            data_frame,
        ) in self.experiment_parameters.test_data_frames_dict.items():
            print(f"testing {label}")
            test_for_this_s_param_combination = ClassificationExperimentLowerLevel()
            result_df = test_data_frame_classifier_frequency_window_with_report(
                data_frame, label, frequency_hop=frequency_hop
            )
            full_results_df = pd.concat((full_results_df, result_df))
        return ClassificationExperimentResults(full_results_df)


class ClassificationExperimentResults:

    def __init__(self, results_df: pd.DataFrame = None):
        self.results_df = results_df
