import numpy as np
from ..definitions.ActivationFunction import ActivationFunction

class Identity(ActivationFunction):
    @staticmethod
    def calculate(xs: np.ndarray[float]) -> np.ndarray[float]:
        return xs
    @staticmethod
    def derivative(xs: np.ndarray[float]) -> np.ndarray[float]:
        return np.ones(len(xs))
