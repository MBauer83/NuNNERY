from src.learning.definitions.Classification import Classification
from .ClassificationMixin import ClassificationMixin

class BinaryClassification(Classification, ClassificationMixin):
    def __init__(self, label_1: str, label_2: str):
        super().__init__([label_1, label_2])