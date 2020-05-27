from chaine.utils import TrainingMetadata


METADATA = TrainingMetadata()


class ConditionalRandomField:
    def __init__(self, squared_sigma: float = 10.0):
        self.squared_sigma = squared_sigma

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    
