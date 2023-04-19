from typing import *
import numpy as np
from learning.definitions import LossFunction
from .LossFunctionMixin import LossFunctionMixin

class MeanSquaredError(LossFunction, LossFunctionMixin):
    @staticmethod
    def calculate(expected: np.ndarray, actual: np.ndarray) -> float:
        return np.sum((expected - actual) ** 2) / len(expected)
    @staticmethod
    def derivative(expected: np.ndarray, actual: np.ndarray) -> np.ndarray:
        return 2 * (actual - expected) / len(expected)
