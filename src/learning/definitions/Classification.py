from typing import List
import numpy as np

class Classification():
    def get_class_names(self) -> List[str]:
        pass
    def get_class_for_one_hot_encoding(self, one_hot_encoding: np.ndarray) -> str:
        pass
    def get_classes_with_values_gte(self, output_activations: np.ndarray, threshold: float) -> List[tuple[str, float]]:
        pass
    def get_classes_with_values_lte(self, output_activations: np.ndarray, threshold: float) -> List[tuple[str, float]]:
        pass
    def get_classes_with_values_gt(self, output_activations: np.ndarray, threshold: float) -> List[tuple[str, float]]:
        pass
    def get_classes_with_values_lt(self, output_activations: np.ndarray, threshold: float) -> List[tuple[str, float]]:
        pass
    def get_one_hot_encoding(self, label: str) -> np.ndarray:
        pass