import pandas as pd

from classification_experiment_parameters import ClassificationExperimentParameters

from feature_extractor import FeatureExtractor


class ClassificationExperimentLowerLevel:
    def __init__(self,):
        pass


# design here is that each classification test has it's own one of these objects,
# will extract features etc for each
class ClassificationExperiment:

    def __init__(self,
                 experiment_parameters: ClassificationExperimentParameters,
                 feature_extractor:FeatureExtractor=None):
        self.experiment_parameters = experiment_parameters
        self.experiment_results = ClassificationExperimentResults()
        self.feature_extractor = feature_extractor


    def run_experiment(self):
        # this is per freq hop -> I think this should be how it works,
        # higher class handles the freq windowing etc
        if self.feature_extractor:
            self.feature_extractor.extract_features()




    #todo what is data_frame? -> I think its s_parameter_data?
    def test_data_frame_classifier_frequency_window_with_report(
            self
            ) -> pd.DataFrame:

        # unpack parameters for some clarity
        s_param_data = self.experiment_parameters.s_param_data

        # as df format is | labels | fq1 | fq2 ......
        # need to get just the fqs which are listed
        freq_list = s_param_data.get_frequency_column_headings_list()

        min_frequency, max_frequency = min(freq_list), max(freq_list)
        low_frequency, high_frequency = min_frequency, min_frequency + self.experiment_parameters.freq_hop

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

    def print_fq_hop(self, high_frequency, label, low_frequency):
        print(f"{label}\n\r{hz_to_ghz(low_frequency)}GHz->{hz_to_ghz(high_frequency)}GHz")


class ClassificationExperimentResults:

    def __init__(self, results_df:pd.DataFrame=None):
        self.results_df = results_df


