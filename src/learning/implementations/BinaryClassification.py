from .ClassificationMixin import ClassificationMixin

class BinaryClassification(ClassificationMixin):
    def __init__(self, label_1: str, label_2: str):
        super().__init__([label_1, label_2])