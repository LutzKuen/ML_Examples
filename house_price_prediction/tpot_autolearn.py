import numpy as np # linear algebra
import code
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np

from tpot import TPOTRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

import os

#  begin data preparation snippet
df_train = pd.read_csv('train.csv')
y = df_train['SalePrice']
x = df_train.drop('SalePrice', axis = 1).drop('Id', axis = 1)
x['TRAIN'] = 1
x_pred = pd.read_csv('test.csv')
id_axis = x_pred['Id'].values[:]
x_pred = x_pred.drop('Id', axis = 1)
x_pred['TRAIN'] = 0
x_full = pd.concat([x, x_pred])
categorical_features = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']
x_ = pd.get_dummies(x_full, columns = categorical_features)
x_t = x_[x_['TRAIN'] == 1]
X_train, X_test, y_train, y_test = train_test_split(x_t.values, y.values,train_size=0.75, test_size=0.25)
# end data preparation snippet
tpot = TPOTRegressor(generations=100, population_size=10, verbosity=2)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_housing_pipeline.py')

#code.interact(banner='', local=locals())

# begin submission snipper
ypred = tpot.predict(x_[x_['TRAIN'] == 0].values)
subfile = open('submission.csv','w')
subfile.write('Id,SalePrice\n')
for i in range(len(ypred)):
 subfile.write(str(int(id_axis[i])) + ',' + str(ypred[i]) + '\n')
subfile.close()
# end submission snippet
