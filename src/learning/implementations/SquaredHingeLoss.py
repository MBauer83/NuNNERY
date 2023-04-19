import numpy as np
from learning.definitions import LossFunction
from .LossFunctionMixin import LossFunctionMixin

class SquaredHingeLoss(LossFunction, LossFunctionMixin):
    @staticmethod
    def calculate(expected: np.ndarray, actual: np.ndarray) -> float:
        return np.sum(np.maximum(0, 1 - expected * actual) ** 2) / len(expected)
    @staticmethod
    def derivative(expected: np.ndarray, actual: np.ndarray) -> np.ndarray:
        return -2 * expected * (1 - expected * actual > 0) * (1 - expected * actual) / len(expected)
