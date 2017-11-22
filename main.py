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
from keras.constraints import  maxnorm
from keras.regularizers import l1_l2


def keras_model(input_size):
    model = Sequential()
    model.add(Dropout(0.2, input_shape=(input_size,)))
    model.add(Dense(200, init='normal', activation='relu', W_constraint=maxnorm(2), W_regularizer=l1_l2(l1=0, l2=1e-4)))
    model.add(Dropout(0.2))
    model.add(Dense(120, init='normal', activation='relu', W_constraint=maxnorm(2), W_regularizer=l1_l2(l1=0, l2=1e-4)))
    model.add(Dropout(0.2))
    model.add(Dense(1, init='normal', activation='sigmoid'))
    # Compile model
    sgd = SGD(lr=0.005, momentum=0.9, decay=0.0, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


data = pd.read_csv("train.csv", sep=',')

data = data.select_dtypes(['number'])
y = data['SalePrice']
x = data.drop(['SalePrice'], axis=1)

x_train, y_train, x_test, y_test = train_test_split(x, y, shuffle=True, train_size=0.8, test_size=0.2)

xgbr = xgb.XGBRegressor()
nn = KerasRegressor(build_fn=keras_model, input_size=x_train.shape[1])
svcr = BaggingRegressor(base_estimator=SVC(), n_estimators=10, max_samples=0.1)

