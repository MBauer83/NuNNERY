from typing import List, Tuple
import numpy as np

class ClassificationMixin():
    def __init__(self, classes: List[str]):
        self._classes = classes
        self._no_of_classes = len(classes)
    def get_class_names(self) -> List[str]:
        return self._classes
    def get_class_for_one_hot_encoding(self, one_hot_encoding: np.ndarray) -> str:
        return self._classes[np.argmax(one_hot_encoding)]
    def get_classes_with_values_gte(self, output_activations: np.ndarray, threshold: float) -> List[Tuple[str, float]]:
        return [(self._classes[i], output_activations[i]) for i in range(len(output_activations)) if output_activations[i] >= threshold]
    def get_classes_with_values_lte(self, output_activations: np.ndarray, threshold: float) -> List[Tuple[str, float]]:
        return [(self._classes[i], output_activations[i]) for i in range(len(output_activations)) if output_activations[i] <= threshold]
    def get_classes_with_values_gt(self, output_activations: np.ndarray, threshold: float) -> List[Tuple[str, float]]:
        return [(self._classes[i], output_activations[i]) for i in range(len(output_activations)) if output_activations[i] > threshold]
    def get_classes_with_values_lt(self, output_activations: np.ndarray, threshold: float) -> List[Tuple[str, float]]:
        return [(self._classes[i], output_activations[i]) for i in range(len(output_activations)) if output_activations[i] < threshold]
    def get_one_hot_encoding(self, label: str) -> np.ndarray:
        one_hot_encoding = np.zeros(self._no_of_classes)
        one_hot_encoding[self._classes.index(label)] = 1