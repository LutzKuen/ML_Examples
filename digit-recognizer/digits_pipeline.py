
import pandas as pd
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import GridSearchCV

# NOTE: Make sure that the class is labeled 'target' in the data file
df_train = pd.read_csv('train.csv')
y = df_train['label']
x = df_train.drop('label', axis = 1)


parameters = {
    'penalty': ['l2'],
    'C': [0.1, 0.5, 1, 2],
    'dual': [True, False]
}


clf = GridSearchCV(LogisticRegression(), parameters, verbose=2)

# Average CV score on the training set was:0.9129525477957177
exported_pipeline = make_pipeline(
    Normalizer(norm="max"),
    clf
)


exported_pipeline.fit(x.values,y.values)
x_pred = pd.read_csv('test.csv')
y_pred = exported_pipeline.predict(x_pred.values)
id_axis = range(1,x_pred.shape[0]+1)
subfile = open('submission.csv', 'w')
subfile.write('ImageId,Label\n')
for _id, label in zip(id_axis, y_pred):
 subfile.write(str(int(_id)) + ',' + str(int(label)) + '\n')
subfile.close()
# end submission snippet


