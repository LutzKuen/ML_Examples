import numpy as np # linear algebra
import code
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np

from tpot import TPOTClassifier
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

import os

#  begin data preparation snippet
df_train = pd.read_csv('train.csv')
y = df_train['label']
x = df_train.drop('label', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(x.values, y.values,train_size=0.75, test_size=0.25)
# end data preparation snippet
tpot = TPOTClassifier(generations=10, population_size=10, verbosity=2)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_digits_pipeline.py')

#code.interact(banner='', local=locals())

# begin submission snipper
x_pred = pd.read_csv('test.csv')
id_axis = range(1,x_pred['Id'].shape[0]+1)
ypred = tpot.predict(x_pred.values)
subfile = open('submission.csv','w')
subfile.write('ImageId,Label\n')
for _id, label in zip(id_axis, label):
 subfile.write(str(int(_id)) + ',' + str(int(label)) + '\n')
subfile.close()
# end submission snippet
