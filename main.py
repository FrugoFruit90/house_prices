import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

data = pd.read_csv("train.csv", sep=',')

data = data.select_dtypes(['number'])
y = data['SalePrice']
x = data.drop(['SalePrice'], axis=1)

x_train, y_train, x_test, y_test = train_test_split(x, y, shuffle=True, train_size=0.8, test_size=0.2)

