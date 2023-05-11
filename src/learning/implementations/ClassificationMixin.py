import numpy as np
from src.learning.definitions.Classification import Classification

class ClassificationMixin(Classification):
    def __init__(self, classes: list[str]):
        self.__classes = classes
        self.__no_of_classes = len(classes)
    def get_class_names(self) -> list[str]:
        return self.__classes
    def get_class_for_one_hot_encoding(self, one_hot_encoding: np.ndarray) -> str:
        return self.__classes[np.argmax(one_hot_encoding)]
    def get_classes_with_values_gte(self, output_activations: np.ndarray, threshold: float) -> list[tuple[str, float]]:
        return [(self.__classes[i], output_activations[i]) for i in range(len(output_activations)) if output_activations[i] >= threshold]
    def get_classes_with_values_lte(self, output_activations: np.ndarray, threshold: float) -> list[tuple[str, float]]:
        return [(self.__classes[i], output_activations[i]) for i in range(len(output_activations)) if output_activations[i] <= threshold]
    def get_classes_with_values_gt(self, output_activations: np.ndarray, threshold: float) -> list[tuple[str, float]]:
        return [(self.__classes[i], output_activations[i]) for i in range(len(output_activations)) if output_activations[i] > threshold]
    def get_classes_with_values_lt(self, output_activations: np.ndarray, threshold: float) -> list[tuple[str, float]]:
        return [(self.__classes[i], output_activations[i]) for i in range(len(output_activations)) if output_activations[i] < threshold]
    def get_one_hot_encoding(self, label: str) -> np.ndarray:
        one_hot_encoding = np.zeros(self.__no_of_classes)
        one_hot_encoding[self.__classes.index(label)] = 1
        return one_hot_encoding