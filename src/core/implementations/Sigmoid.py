import numpy as np
from src.core.definitions.ActivationFunction import ActivationFunction

class Sigmoid(ActivationFunction):
    @staticmethod
    def calculate(xs: np.ndarray[float]) -> np.ndarray[float]:
        return 1 / (1 + np.exp(-xs))
    @staticmethod
    def derivative(xs: np.ndarray[float]) -> np.ndarray[float]:
        sig = Sigmoid.calculate(xs)
        return np.multiply(sig,(np.ones(len(xs)) - Sigmoid.calculate(xs)))
