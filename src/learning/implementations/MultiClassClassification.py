
from src.learning.definitions.Classification import Classification
from .ClassificationMixin import ClassificationMixin

class MultiClassClassification(Classification, ClassificationMixin):
    def __init__(self, labels: list[str]):
        super().__init__(labels)