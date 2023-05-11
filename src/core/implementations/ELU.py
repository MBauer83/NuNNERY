import numpy as np
from ..definitions.ActivationFunction import ActivationFunction

class ELU(ActivationFunction):
    @staticmethod
    def calculate(xs: np.ndarray[float]) -> np.ndarray[float]:
        return np.where(xs > 0, xs, 1.0 * (np.exp(xs) - 1))
    @staticmethod
    def derivative(xs: np.ndarray[float]) -> np.ndarray[float]:
        return np.where(xs > 0, np.ones(len(xs)), np.exp(xs))
