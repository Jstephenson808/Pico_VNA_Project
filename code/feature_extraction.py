from abc import ABC, abstractmethod

from classification_test import ClassificationExperiment, ClassificationExperimentParameters


class FeatureExtractor(ABC):

    @abstractmethod
    def extract_features(self):
        pass


class TsFreshFeatureExtractor(FeatureExtractor):

    def __init__(self,
                 experiment_parameters: ClassificationExperimentParameters,
                 ):
