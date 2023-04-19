import numpy as np
from core.definitions import ActivationFunction

class Softmax(ActivationFunction):
    @staticmethod
    def calculate(xs: np.ndarray[float]) -> np.ndarray[float]:
        exps = np.exp(xs - np.max(xs))
        return exps / np.sum(exps)
    @staticmethod
    def derivative(xs: np.ndarray[float]) -> np.ndarray[float]:
        return Softmax.calculate(xs) * (1 - Softmax.calculate(xs))