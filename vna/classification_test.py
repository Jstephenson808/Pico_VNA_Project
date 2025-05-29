from __future__ import annotations

from abc import ABC, abstractclassmethod, abstractmethod

import pandas as pd
from sklearn.pipeline import Pipeline

from classification_experiment_parameters import (
    FrequencyHopClassificationExperimentParameters,
)
from s_parameter_data import SParameterData
from frequency import Frequency

from feature_extractor import FeatureExtractor
from vna.classifier import Classifier
from vna.feature_extractor import ExtractedFeatures
from vna.movement_vector import MovementVector


class ClassificationTestStratergy(ABC):

    @abstractmethod
    def run_classification(self) -> ClassificationExperimentResults:
        pass


class FrequencyHopClassification:
    def __init__(
        self,
        s_param_data_under_test: SParameterData,
        test_label: str,
        frequency_hop: Frequency,
        classifiers_to_test: [Classifier],
        movement_vector: MovementVector,
        feature_extractor: FeatureExtractor | None = None,
        extracted_features: ExtractedFeatures | None = None,
    ):
        self.s_param_data_under_test: SParameterData = s_param_data_under_test
        self.test_label: str = test_label
        self.frequency_hop: Frequency = frequency_hop
        self.feature_extractor: FeatureExtractor = feature_extractor
        self.classifiers_to_test: list[Classifier] = classifiers_to_test
        self.extracted_features: ExtractedFeatures = extracted_features
        self.movement_vector: MovementVector = movement_vector
        self.test_minimum_frequency: Frequency = (
            self.s_param_data_under_test.get_minimum_frequency()
        )
        self.test_maximum_frequency: Frequency = (
            self.s_param_data_under_test.get_maximum_frequency()
        )

        # todo this needs to be in a lower class for experiment

    def test_data_frame_classifier_frequency_window_with_report(self) -> pd.DataFrame:
        """
        This is a copy over from the previous implementation
        Handles all testing and classification, feels like this should be a few more methods really
        Returns:

        """

        low_frequency: Frequency = self.test_minimum_frequency

        # jump by frequency hop each time
        high_frequency: Frequency = self.calculate_high_frequency(low_frequency)

        while high_frequency <= self.test_maximum_frequency:
            self.print_fq_hop(high_frequency, self.test_label, low_frequency)

            try:
                data_frame_fq_range_filtered = (
                    self.s_param_data_under_test.get_data_frame_between_frequency(
                        low_frequency, high_frequency
                    )
                )
                # why is this here?
            except ValueError as e:
                print(e)
                continue

            # This fixes the column titles for feature extraction purposes
            # Should refactor so this does not need to happen or document *why*
            data_frame_fq_range_filtered.make_columns_have_s_param_mag_phase_titles()

            # label for this fq band and test
            fq_label = self.generate_classification_test_label(
                low_frequency, high_frequency
            )

            # if there is a passed feature extractor and
            if self.feature_extractor and (self.extracted_features is None):
                # extract features from time series
                self.extracted_features = self.feature_extractor.extract_features(
                    data_frame_fq_range_filtered
                )

            # now need to do the test
            for classifier in self.classifiers_to_test:
                classifier.run_classifier(self.extracted_features, self.movement_vector)

    def calculate_high_frequency(self, low_frequency: Frequency):
        return low_frequency + self.frequency_hop

    def generate_classification_test_label(
        self, low_frequency: Frequency, high_frequency: Frequency
    ):
        return f"{self.test_label}_{low_frequency.get_freq_ghz()}GHz_{high_frequency.get_freq_ghz()}GHz"

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
        experiment_parameters: FrequencyHopClassificationExperimentParameters,
        pipeline: Pipeline,
    ):
        self.experiment_parameters: FrequencyHopClassificationExperimentParameters = (
            experiment_parameters
        )
        self.experiment_results: ClassificationExperimentResults = None
        self.pipeline: Pipeline = pipeline

    def run_experiment(self):
        # this is per freq hop -> I think this should be how it works,
        # higher class handles the freq windowing etc

        # first need to split data according to fq/s_param plan

        # extract features

        return

    def test_classifier_from_df_dict(self) -> ClassificationExperimentResults:
        """
        This returns a report and save classifier to pkl path
        """
        full_results_df = None
        for (
            label,
            s_param_data_under_test,
        ) in self.experiment_parameters.test_data_frames_dict.items():
            print(f"testing {label}")
            classification_for_this_test = FrequencyHopClassification(
                s_param_data_under_test=s_param_data_under_test,
                test_label=label,
                frequency_hop=self.experiment_parameters.freq_hop,
                feature_extractor=self.feature_extractor,
                movement_vector=self.experiment_parameters.movement_vector,
            )

        return ClassificationExperimentResults(full_results_df)
