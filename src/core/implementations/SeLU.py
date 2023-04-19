import numpy as np
from src.core.definitions.ActivationFunction import ActivationFunction

class SeLU(ActivationFunction):
    @staticmethod
    def calculate(xs: np.ndarray[float]) -> np.ndarray[float]:
        return np.where(xs > 0, xs, 1.6732632423543772848170429916717 * (np.exp(xs) - 1))
    @staticmethod
    def derivative(xs: np.ndarray[float]) -> np.ndarray[float]:
        return np.where(xs > 0, np.ones(len(xs)), 1.6732632423543772848170429916717 * np.exp(xs))
