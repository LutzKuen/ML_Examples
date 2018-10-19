#
# Keras model to compute the digit recognition example from kaggle
#
from keras.models import Sequential
from keras.layers import Dense, Activation
import pandas as pd
from keras.utils import to_categorical
import numpy as np
import code
# prepare keras
model = Sequential([
    Dense(32, input_shape=(784,)),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# read the data
df_train = pd.read_csv('train.csv')
y = to_categorical(df_train['label'].values)
x = np.array(df_train.drop('label', axis = 1))
# do the thing
#code.interact(local=locals())
model.fit(x, y, epochs=10)#, batch_size=32)

x_pred = np.array(pd.read_csv('test.csv'))
y_pred = model.predict(x_pred)
y_pred = [np.argmax(y, axis=None, out=None) for y in y_pred]
id_axis = range(1,x_pred.shape[0]+1)
subfile = open('submission_keras.csv', 'w')
subfile.write('ImageId,Label\n')
for _id, label in zip(id_axis, y_pred):
 subfile.write(str(int(_id)) + ',' + str(int(label)) + '\n')
subfile.close()
