import numpy as np
from src.learning.definitions import LossFunction
from .LossFunctionMixin import LossFunctionMixin

class SparseCategoricalCrossEntropy(LossFunction, LossFunctionMixin):
    @staticmethod
    def calculate(expected: np.ndarray, actual: np.ndarray) -> float:
        return -np.sum(np.log(actual[expected == 1])) / len(expected)
    @staticmethod
    def derivative(expected: np.ndarray, actual: np.ndarray) -> np.ndarray:
        return -expected / actual / len(expected)
