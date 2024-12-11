from abc import ABC, abstractmethod

from sklearn.tree import DecisionTreeClassifier


class Classifier(ABC):

    @abstractmethod
    def run_classifier(self):
        pass


class Pico_DecisionTreeClassifier(Classifier):
    def __init__(self, classifier: DecisionTreeClassifier = DecisionTreeClassifier()):
        self.classifier = classifier
