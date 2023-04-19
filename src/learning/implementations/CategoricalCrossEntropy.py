import numpy as np
from learning.definitions import LossFunction
from .LossFunctionMixin import LossFunctionMixin

class CategoricalCrossEntropy(LossFunction, LossFunctionMixin):
    @staticmethod
    def calculate(expected: np.ndarray, actual: np.ndarray) -> float:
        # Avoid taking the log of 0
        epsilon = 1e-15
        # Clip the actual values to avoid taking the log of values close to 0
        actual = np.clip(actual, epsilon, 1 - epsilon)
        return -np.sum(expected * np.log(actual))
    @staticmethod
    def derivative(expected: np.ndarray, actual: np.ndarray) -> np.ndarray:
        # Avoid taking the log of 0
        epsilon = 1e-15
        # Clip the actual values to avoid taking the log of values close to 0
        actual = np.clip(actual, epsilon, 1 - epsilon)
        return actual - expected
