import numpy as np
from src.core.definitions.ActivationFunction import ActivationFunction

class ReLU(ActivationFunction):
    @staticmethod
    def calculate(xs: np.ndarray[float]) -> np.ndarray[float]:
        return np.maximum(0, xs)
    @staticmethod
    def derivative(xs: np.ndarray[float]) -> np.ndarray[float]:
        return 1. * (xs > 0)
