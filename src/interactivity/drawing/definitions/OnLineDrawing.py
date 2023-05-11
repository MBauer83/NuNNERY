import numpy as np
from abc import ABC, abstractmethod
from typing import Generator
import sys
sys.path.insert(0, "..")
from ..value_objects.LineDrawingConfiguration import LineDrawingConfiguration


class OnLineDrawing(ABC):
    @abstractmethod
    def run(self, config: LineDrawingConfiguration) -> Generator[np.ndarray, None, None]:
        raise NotImplementedError
    
    @classmethod
    def __subclasshook__(cls, subclass):
        return ((hasattr(subclass, 'run') and
                callable(subclass.run)) or
                NotImplemented)