# model_base.py

from abc import ABC, abstractmethod

class ObjectDetectionModel(ABC):
    @abstractmethod
    def load_model(self, model_path: str):
        pass

    @abstractmethod
    def predict(self, image):
        pass
