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
        return pd.DataFrame(self.model.predict(X))  # change predict_proba to predict if you want predictions
