# -*- coding: utf-8 -*-
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline, FeatureUnion
from keras.models import Sequential
from keras.layers import Dropout, Dense
from keras.optimizers import SGD
from keras.constraints import maxnorm
from keras.regularizers import l1_l2
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.preprocessing import StandardScaler
from utils import ModelTransformer, make_keras_picklable
from sklearn.linear_model import LinearRegression
import dill as pickle


def keras_model(input_size):
    make_keras_picklable()
    model = Sequential()
    model.add(Dense(200, input_shape=(input_size,), kernel_initializer='normal', activation='relu',
                    kernel_constraint=maxnorm(2),
                    kernel_regularizer=l1_l2(l1=0, l2=1e-4)))
    model.add(Dropout(0.2))
    model.add(Dense(120, kernel_initializer='normal', activation='relu', kernel_constraint=maxnorm(2),
                    kernel_regularizer=l1_l2(l1=0, l2=1e-4)))
    model.add(Dropout(0.2))
    model.add(Dense(1, kernel_initializer="normal"))
    # Compile model
    model.compile(loss='mse', optimizer='adam')
    return model


data = pd.read_csv("train.csv", sep=',')

data = data.select_dtypes(['number']).dropna(axis=1, how='any', thresh=None, subset=None, inplace=False)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

print(data.shape)
y = scaled_data[:, -1]
x = scaled_data[:, :-1]

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, train_size=0.8, test_size=0.2)

xgbr = xgb.XGBRegressor()
nn = KerasRegressor(build_fn=keras_model, input_size=x_train.shape[1])
svcr = BaggingRegressor(base_estimator=SVR(), n_estimators=10, max_samples=0.1)

pipeline = Pipeline([
    ('models', FeatureUnion([('xgb', ModelTransformer(xgbr)),
                             ('nn', ModelTransformer(nn)),
                             ('svm', ModelTransformer(svcr))])),
    ('voting', LinearRegression())

])
pipeline.fit(x_train, y_train)
y_pred_pipeline = pipeline.predict(x_test)
print(mean_squared_error(y_pred_pipeline, y_test))

s = pickle.dumps(pipeline)
clf2 = pickle.loads(s)

y_pred_pickle = clf2.predict(x_test)
print(mean_squared_error(y_pred_pickle, y_test))

# xgbr.fit(x_train, y_train)
# nn.fit(x_train, y_train)
# svcr.fit(x_train, y_train)
#
# y_pred_xgb = xgbr.predict(x_test)
# y_pred_nn = nn.predict(x_test)
# y_pred_svcr = svcr.predict(x_test)
#
# print(mean_squared_error(y_pred_xgb, y_test))
# print(mean_squared_error(y_pred_nn, y_test))
# print(mean_squared_error(y_pred_svcr, y_test))
