import numpy as np
from src.learning.definitions import LossFunction
from .LossFunctionMixin import LossFunctionMixin

class KullbackLeiblerDivergence(LossFunction, LossFunctionMixin):
    @staticmethod
    def calculate(expected: np.ndarray, actual: np.ndarray) -> float:
        return np.sum(expected * np.log(expected / actual)) / len(expected)
    @staticmethod
    def derivative(expected: np.ndarray, actual: np.ndarray) -> np.ndarray:
        return (np.log(expected / actual) - 1) / len(expected)