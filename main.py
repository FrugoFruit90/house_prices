import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from keras.models import Sequential
from keras.layers import Dropout, Dense
from keras.optimizers import SGD
from keras.constraints import maxnorm
from keras.regularizers import l1_l2
from sklearn.metrics import mean_squared_error
import numpy as np


def keras_model(input_size):
    model = Sequential()
    model.add(Dense(200, input_shape=(input_size,), kernel_initializer='normal', activation='relu', W_constraint=maxnorm(2),
                    kernel_regularizer=l1_l2(l1=0, l2=1e-4)))
    model.add(Dropout(0.2))
    model.add(Dense(120, init='normal', activation='relu', W_constraint=maxnorm(2), W_regularizer=l1_l2(l1=0, l2=1e-4)))
    model.add(Dropout(0.2))
    model.add(Dense(1, init='normal', activation='relu'))
    # Compile model
    sgd = SGD(lr=0.005, momentum=0.9, decay=0.0, nesterov=True)
    model.compile(loss='mse', optimizer=sgd)
    return model


data = pd.read_csv("train.csv", sep=',')

data = data.select_dtypes(['number'])
y = data['SalePrice']
x = data.drop(['SalePrice'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, train_size=0.8, test_size=0.2)

xgbr = xgb.XGBRegressor()
nn = KerasRegressor(build_fn=keras_model, input_size=x_train.shape[1])
svcr = BaggingRegressor(base_estimator=SVC(), n_estimators=10, max_samples=0.1)

xgbr.fit(x_train, np.array(y_train))
nn.fit(x_train, y_train)
svcr.fit(x_train, y_train)

y_pred_xgb = xgbr.predict(x_test)
y_pred_nn = nn.predict(x_test)
y_pred_svcr = svcr.predict(x_test)

print(mean_squared_error(y_pred_xgb, y_test))
print(mean_squared_error(y_pred_nn, y_test))
print(mean_squared_error(y_pred_svcr, y_test))
