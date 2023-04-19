import numpy as np
from learning.definitions import LossFunction
from .LossFunctionMixin import LossFunctionMixin

class HingeLoss(LossFunction, LossFunctionMixin):
    @staticmethod
    def calculate(expected: np.ndarray, actual: np.ndarray) -> float:
        return np.sum(np.maximum(0, 1 - expected * actual)) / len(expected)
    @staticmethod
    def derivative(expected: np.ndarray, actual: np.ndarray) -> np.ndarray:
        return -expected * (1 - expected * actual > 0) / len(expected)
