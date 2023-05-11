from .ClassificationMixin import ClassificationMixin

class MultiClassClassification(ClassificationMixin):
    def __init__(self, labels: list[str]):
        super().__init__(labels)