import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer

from sklearn.model_selection import GridSearchCV

# NOTE: Make sure that the class is labeled 'target' in the data file
#tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
#features = tpot_data.drop('target', axis=1).values
#training_features, testing_features, training_target, testing_target = \
#            train_test_split(features, tpot_data['target'].values, random_state=None)


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

imputer = Imputer(strategy="median")
imputer.fit(x_.values)
training_features = imputer.transform(x_t.values)
testing_features = imputer.transform(x_[x_['TRAIN'] == 0].values)

# Average CV score on the training set was:-580352114.3134693
parameters = {
        'alpha': [0.99],
        'learning_rate': [0.01, 0.1, 0.25],
        'loss': ['ls'],
        'max_depth': [6, 24, 48],
        'max_features': ['auto', 'sqrt', 'log2' ],
        'n_estimators': [100, 500, 1000 ],
        'subsample': [ 0.75, 1.0] }
clf = GridSearchCV(GradientBoostingRegressor(), parameters, verbose = 2)
exported_pipeline = make_pipeline(
    SelectPercentile(score_func=f_regression, percentile=91),
    #GradientBoostingRegressor(alpha=0.99, learning_rate=0.1, loss="ls", max_depth=6, max_features=1.0, min_samples_leaf=1, min_samples_split=17, n_estimators=100, subsample=1.0)
    clf
)

exported_pipeline.fit(training_features, y.values)
ypred = exported_pipeline.predict(testing_features)

# begin submission snipper
subfile = open('submission.csv','w')
subfile.write('Id,SalePrice\n')
for i in range(len(ypred)):
     subfile.write(str(int(id_axis[i])) + ',' + str(ypred[i]) + '\n')
subfile.close()
# end submission snippet
print(clf.best_parameters_)
