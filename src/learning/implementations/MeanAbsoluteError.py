import numpy as np
from src.learning.definitions import LossFunction
from .LossFunctionMixin import LossFunctionMixin

class MeanAbsoluteError(LossFunction, LossFunctionMixin):
    @staticmethod
    def calculate(expected: np.ndarray, actual: np.ndarray) -> float:
        return np.sum(np.abs(expected - actual)) / len(expected)
    @staticmethod
    def derivative(expected: np.ndarray, actual: np.ndarray) -> np.ndarray:
        return np.sign(actual - expected) / len(expected)
