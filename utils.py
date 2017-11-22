from sklearn.base import TransformerMixin, BaseEstimator, ClassifierMixin
import pandas as pd
import numpy as np
from scipy.stats.mstats import gmean, hmean
from sklearn.preprocessing import binarize


class ModelTransformer(TransformerMixin):
    def __init__(self, model):
        self.model = model

    def fit(self, *args, **kwargs):
        self.model.fit(*args, **kwargs)
        return self

    def transform(self, X, **transform_params):
        return pd.DataFrame(self.model.predict_proba(X))  # change predict_proba to predict if you want predictions


class EnsembleBinaryClassifier(BaseEstimator, ClassifierMixin, TransformerMixin):
    """ Average or majority-vote several different classifiers.
    Assumes input is a matrix of individual predictions, such as the output of a FeatureUnion of ModelTransformers
    [n_samples, n_predictors]."""

    def __init__(self, mode, weights=None, threshold=0.5, mean_calc="avg"):
        self.mode = mode
        self.weights = weights
        self.threshold = threshold
        self.mean_calc = mean_calc

    def fit(self, X, y=None):
        return self

    def predict_proba(self, X):
        """"Predict (weighted) probabilities"""
        X_new = pd.DataFrame(X)
        X_new = X_new[X_new.columns[::2]]
        if self.mean_calc == "avg":
            probs = np.average(X_new, axis=1, weights=self.weights)
        elif self.mean_calc == "rms":
            probs = np.sqrt(np.average(X_new ** 2, axis=1))
        elif self.mean_calc == "geom":
            probs = gmean(X_new, axis=1)
        elif self.mean_calc == "harmonic":
            probs = hmean(X_new + 0.00001, axis=1)  # all elements need to be greater than 0

        # return probs
        # return np.column_stack((1 - probs, probs))
        return np.column_stack((1 - probs, probs))

    def predict(self, X):
        """"Predict class labels."""
        if self.mode == 'average':
            return binarize(self.predict_proba(X)[:, [1]], self.threshold)
        else:
            res = binarize(X, 0.5)
            return np.apply_along_axis(lambda x: np.bincount(x.astype(int), self.weights).argmax(), axis=1, arr=res)
