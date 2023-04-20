import numpy as np
from src.learning.definitions import LossFunction
from .LossFunctionMixin import LossFunctionMixin

class BinaryCrossEntropy(LossFunction, LossFunctionMixin):
    @staticmethod
    def calculate(expected: np.ndarray, actual: np.ndarray) -> float:
        return -np.sum(expected * np.log(actual) + (1 - expected) * np.log(1 - actual)) / len(expected)
    @staticmethod
    def derivative(expected: np.ndarray, actual: np.ndarray) -> np.ndarray:
        return (actual - expected) / (actual * (1 - actual))
