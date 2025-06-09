from abc import ABC, abstractmethod

import numpy as np
from numpy.random import RandomState
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from vna.VNA_defaults import DEFAULT_CLASSIFICATION_TRAIN_TEST_SPLIT
from vna.feature_extractor import ExtractedFeatures
from vna.movement_vector import MovementVectorPandas


class ClassificationResults(ABC):

    @abstractmethod
    def process_test_results(self, *, test_results, test_labels):
        pass


class SkLearnClassificationResults:
    """
    Holds the results of the classification, the report generated as well as confusion matrix
    """

    def __init__(self):
        self.classification_test_results = None
        self.classification_test_report = None
        self.confusion_matrix = None

    def process_test_results(self, *, test_results, test_labels):
        self.classification_test_results = test_results
        self.classification_test_report = classification_report(
            test_labels, test_results, output_dict=True
        )
        self.confusion_matrix = confusion_matrix(test_labels, test_results)


class Classifier(ABC):
    """
    All classifiers must implement this interface.

    The design is such that the classifiers hold all the parameters for the classifaction
    algorithm and the data for it is injected by the caller. This allows for the same
    classifier to be reused
    """

    @abstractmethod
    def run_classifier(
        self,
        extracted_features: ExtractedFeatures,
        movement_vector: MovementVectorPandas,
    ):
        pass


class PicoDecisionTreeClassifier(Classifier):
    """ """

    def __init__(
        self,
        *,
        decision_tree_classifier: DecisionTreeClassifier = DecisionTreeClassifier(),
        random_state: RandomState = None,
        train_test_split=DEFAULT_CLASSIFICATION_TRAIN_TEST_SPLIT,
        scaler: StandardScaler = StandardScaler(),
    ):
        self.classifier = decision_tree_classifier
        if random_state is None:
            random_state = np.random.RandomState()
        self.random_state = random_state
        self.classification_results: SkLearnClassificationResults = (
            SkLearnClassificationResults()
        )
        self.train_test_split = train_test_split
        self.scaler = scaler

    def normailse_data(self, extracted_features: ExtractedFeatures):
        self.scaler.fit(extracted_features.extracted_features)
        return self.scaler.transform(extracted_features.extracted_features)

    def run_classifier(
        self,
        extracted_features: ExtractedFeatures,
        movement_vector: MovementVectorPandas,
    ):
        normailsed_data = self.normailse_data(extracted_features)
        training_data, test_data, training_labels, test_labels = train_test_split(
            normailsed_data,
            movement_vector.movement_vector,
            test_size=self.train_test_split,
            random_state=self.random_state,
        )

        self.classifier = self.classifier.fit(training_data, training_labels)
        test_results = self.classifier.predict(test_data, test_labels)
        self.classification_results.process_test_results(
            test_results=test_results, test_labels=test_labels
        )


class SupportVectorClassifier(Classifier):
    def __init__(
        self,
        *,
        svc_classifier: SVC = SVC(),
        scaler: StandardScaler = StandardScaler(),
        random_state: RandomState = None,
        train_test_split=0.4,
    ):
        self.classifier: SVC = svc_classifier
        self.classification_results: SkLearnClassificationResults = (
            SkLearnClassificationResults()
        )
        self.scaler = scaler
        self.random_state = random_state
        self.train_test_split = train_test_split

    def normailse_data(self, extracted_features: ExtractedFeatures):
        self.scaler.fit(extracted_features.extracted_features)
        return self.scaler.transform(extracted_features.extracted_features)

    def run_classifier(
        self,
        extracted_features: ExtractedFeatures,
        movement_vector: MovementVectorPandas,
    ):
        normailsed_features = self.normailse_data(extracted_features)

        training_data, test_data, training_labels, test_labels = train_test_split(
            normailsed_features,
            movement_vector.movement_vector,
            test_size=0.4,
            random_state=self.random_state,
        )

        self.classifier: SVC = self.classifier.fit(training_data, training_labels)
        test_results = self.classifier.predict(test_data, test_labels)
        self.classification_results.process_test_results(
            test_results=test_results, test_labels=test_labels
        )
