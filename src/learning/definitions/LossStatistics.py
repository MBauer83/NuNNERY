import numpy as np

class LossStatistics:
    def get_sum_losses(self) -> float:
        pass
    def get_mean_loss(self) -> float:
        pass
    def print(self):
        pass
    @staticmethod
    def from_float_array(array: np.ndarray[float]) -> 'LossStatistics':
        pass
    @staticmethod
    def from_statistics(array: np.ndarray['LossStatistics']) -> 'LossStatistics':
        pass