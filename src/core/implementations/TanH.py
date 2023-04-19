import numpy as np
from src.core.definitions.ActivationFunction import ActivationFunction

class TanH(ActivationFunction):
    @staticmethod
    def calculate(xs: np.ndarray[float]) -> np.ndarray[float]:
        return np.tanh(xs)
    @staticmethod
    def derivative(xs: np.ndarray[float]) -> np.ndarray[float]:
        return 1 - np.square(np.tanh(xs))
    