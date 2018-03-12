#!/usr/bin/env python

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from keras.layers import Conv2D, MaxPooling2D, Dense, Reshape, Flatten, Input
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
import numpy as np
from read_data import read_data

### MODELO
def cnn():
    model = Sequential()

    #model.add(Reshape((28, 28, 1), input_shape = (28, 28)))
    model.add(Conv2D(
        input_shape = (160, 120, 3)
        , filters = 32
        , kernel_size = (3, 3)
        , padding = "same"
        , activation = "relu"
    ))
    model.add(MaxPooling2D())
    model.add(Conv2D(
        filters = 64
        , kernel_size = (3, 3)
        , padding = "same"
        , activation = "relu"
    ))
    model.add(MaxPooling2D())
    model.add(Conv2D(
        filters = 128
        , kernel_size = (3, 3)
        , padding = "same"
        , activation = "relu"
    ))
    model.add(Flatten())
    model.add(Dense(2048))
    model.add(Dense(100))
    model.add(Dense(2, activation = "softmax"))

    return model

def dense():
    model = Sequential()

#### TRAIN
x_train, y_train, x_test = read_data()
        
# normalize
#x_train = x_train.astype('float32') / 255.
#x_test = x_test.astype('float32') / 255.

y_train = to_categorical(y_train, num_classes = 2)
#y_test = to_categorical(y_test, num_classes = 10)

print(x_train.shape)
print(y_train.shape)

model = cnn()

model.compile("rmsprop", "categorical_crossentropy")
model.fit(
    x_train, y_train,
    batch_size = 32,
    epochs = 2,
    shuffle = True
)

preds = model.predict(x_test)
print(preds)
