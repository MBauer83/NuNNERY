import numpy as np
from learning.definitions import LossFunction
from .LossFunctionMixin import LossFunctionMixin

class MeanSquaredLogError(LossFunction, LossFunctionMixin):
    @staticmethod
    def calculate(expected: np.ndarray, actual: np.ndarray) -> float:
        return np.sum((np.log(expected + 1) - np.log(actual + 1)) ** 2) / len(expected)
    @staticmethod
    def derivative(expected: np.ndarray, actual: np.ndarray) -> np.ndarray:
        return 2 * (np.log(actual + 1) - np.log(expected + 1)) / len(expected)
