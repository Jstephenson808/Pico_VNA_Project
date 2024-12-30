from abc import ABC, abstractmethod

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from vna.feature_extractor import ExtractedFeatures
from vna.movement_vector import MovementVector

class ClassificationResults:
    def __init__(self):
        self.classification_test_results = None
        self.classification_test_report = None
        self.confusion_matrix = None

    def process_test_results(self, *, test_results, test_labels):
        self.classification_test_results = test_results
        self.classification_test_report = classification_report(test_labels, test_results, output_dict=True)
        self.confusion_matrix = confusion_matrix(test_labels, test_results)

class Classifier(ABC):

    @abstractmethod
    def run_classifier(self):
        pass


class PicoDecisionTreeClassifier(Classifier):
    """
    By passing in the classifier object the parameters of the classification can be tweaked
    """
    def __init__(self, *, classifier: DecisionTreeClassifier = DecisionTreeClassifier(), full_data_set: ExtractedFeatures, movement_vector: MovementVector):
        self.classifier = classifier
        self.training_data = full_data_set
        self.movement_vector = movement_vector
        self.classification_results: ClassificationResults = ClassificationResults()

    def run_classifier(self):
        training_data, test_data, training_labels, test_labels = train_test_split(
            self.training_data.extracted_features, self.movement_vector.movement_vector, test_size=0.4
        )

        self.classifier = self.classifier.fit(training_data, training_labels)
        test_results = self.classifier.predict(test_data, test_labels)
        self.classification_results.process_test_results(test_results=test_results, test_labels=test_labels)



