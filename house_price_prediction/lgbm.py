import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import lightgbm as lgb
import numpy as np

import os


df_train = pd.read_csv('train.csv')

y = df_train['SalePrice']

x = df_train.drop('SalePrice', axis = 1).drop('Id', axis = 1)

categorical_features = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']
x_ = pd.get_dummies(x, columns = categorical_features)
feature_name = x_.columns
data_full = lgb.dataset(x_,y)

lgb_params = dict(
            objective = 'regression_l1',
                learning_rate = 0.01,
                    num_iterations = 10000,
                        num_leaves = 1200
                        )

model = lgb.train(lgb_params, data_full)

x_pred = pd.read_csv('../input/test.csv')

id_axis = x_pred['Id'].values[:]

x_pred = x_pred.drop('Id', axis = 1)

x_pred_ = pd.get_dummies(x_pred, columns = categorical_features)

ypred = model.predict(x_pred_)

print(ypred.shape)
print(id_axis.shape)


subfile = open('submission.csv','w')
subfile.write('Id,SalePrice\n')
for i in range(len(ypred)):
 subfile.write(str(int(id_axis[i])) + ',' + str(ypred[i]) + '\n')
 subfile.close()
