import numpy as np
from core.definitions import ActivationFunction

class LeakyReLU(ActivationFunction):
    @staticmethod
    def calculate(xs: np.ndarray[float]) -> np.ndarray[float]:
        return np.maximum(0.01 * xs, xs)
    @staticmethod
    def derivative(xs: np.ndarray[float]) -> np.ndarray[float]:
        return 1. * (xs > 0) + 0.01 * (xs <= 0)
