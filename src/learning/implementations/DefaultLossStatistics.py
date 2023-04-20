import numpy as np
from src.learning.definitions.LossStatistics import LossStatistics

class DefaultLossStatistics(LossStatistics):
    def __init__(self, sum_losses: float, mean_loss: float):
        self.sum_losses = sum_losses
        self.mean_loss = mean_loss
    def get_sum_losses(self) -> float:
        return self.sum_losses
    def get_mean_loss(self) -> float:
        return self.mean_loss
    def print(self):
        print("Sum losses: " + str(self.sum_losses))
        print("Mean loss: " + str(self.mean_loss))
    @staticmethod
    def from_float_array(array: np.ndarray[float]) -> 'DefaultLossStatistics':
        return DefaultLossStatistics(np.sum(array), np.mean(array))
    @staticmethod
    def from_statistics(array: np.ndarray['DefaultLossStatistics']) -> 'DefaultLossStatistics':
        array = array.flatten()
        sum_losses = np.sum([s.get_sum_losses() for s in array])
        mean_loss = np.mean([s.get_mean_loss() for s in array])
        return DefaultLossStatistics(sum_losses, mean_loss)