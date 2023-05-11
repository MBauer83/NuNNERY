import numpy as np
from ..definitions.ActivationFunction import ActivationFunction

class SeLU(ActivationFunction):
    ALPHA = 1.6732632423543772848170429916717
    LAMBDA = 1.0507009873554804934193349852946
    @staticmethod
    def calculate(xs: np.ndarray[float]) -> np.ndarray[float]:
        return SeLU.LAMBDA * np.where(xs > 0, xs, SeLU.ALPHA * (np.exp(xs) - 1))
    @staticmethod
    def derivative(xs: np.ndarray[float]) -> np.ndarray[float]:
        return SeLU.LAMBDA * np.where(xs > 0, np.ones(len(xs)), SeLU.ALPHA * np.exp(xs))
