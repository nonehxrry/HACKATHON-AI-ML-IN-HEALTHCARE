# my_module.py

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# Example of a custom transformer
class CustomTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, parameter=1):
        self.parameter = parameter

    def fit(self, X, y=None):
        # Custom fit logic
        return self

    def transform(self, X, y=None):
        # Custom transform logic
        return X * self.parameter

# Example of a custom model
class CustomModel:
    def __init__(self, n_estimators=10):
        self.n_estimators = n_estimators

    def fit(self, X, y):
        # Custom fit logic
        pass

    def predict(self, X):
        # Custom predict logic
        return np.zeros(len(X))

# Example utility function
def custom_function(x):
    return x * 2
