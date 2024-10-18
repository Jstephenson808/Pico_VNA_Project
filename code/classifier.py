from abc import ABC, abstractmethod


class Classifier(ABC):

    @abstractmethod
    def run_classifier(self):
        pass

class