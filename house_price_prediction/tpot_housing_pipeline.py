import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=None)

imputer = Imputer(strategy="median")
imputer.fit(training_features)
training_features = imputer.transform(training_features)
testing_features = imputer.transform(testing_features)

# Average CV score on the training set was:-580352114.3134693
exported_pipeline = make_pipeline(
    SelectPercentile(score_func=f_regression, percentile=91),
    GradientBoostingRegressor(alpha=0.99, learning_rate=0.1, loss="ls", max_depth=6, max_features=1.0, min_samples_leaf=1, min_samples_split=17, n_estimators=100, subsample=1.0)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
