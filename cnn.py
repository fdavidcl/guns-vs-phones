#!/usr/bin/env python
import numpy as np
np.random.seed(1234) # for reproducibility

# https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
import os

os.environ['PYTHONHASHSEED'] = '0'
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

from keras.layers import Conv2D, MaxPooling2D, Dense, Reshape, Flatten, Input, Dropout, GlobalAveragePooling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.models import Sequential, Model
from keras.utils.np_utils import to_categorical
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
import keras.applications as ka
import numpy as np
from read_data import read_data
from write_predictions import write_predictions

### MODELO
def cnn():
    model = Sequential()

    #model.add(Reshape((28, 28, 1), input_shape = (28, 28)))
    model.add(Conv2D(
        input_shape = (200, 200, 3)
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

def use_base_model(base_model_f):
    base_model = base_model_f(weights = 'imagenet', include_top = False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation = 'relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation = 'relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(2, activation = 'softmax')(x)

    model = Model(base_model.input, outputs = x)

    for layer in base_model.layers:
        layer.trainable = False

    return model

def vgg16():
    base_model = VGG16(weights = 'imagenet', include_top = False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation = 'relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation = 'relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(2, activation = 'softmax')(x)

    model = Model(base_model.input, outputs = x)

    for layer in base_model.layers:
        layer.trainable = False

    return model

def vgg16NOT():
    model = VGG16(
        include_top = False,
        weights = 'imagenet',
        input_shape = (160, 120, 3),
        classes = 2
    )

    return model


def __vgg16(weights_path='vgg16_weights.h5'):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='relu'))

    if weights_path:
        model.load_weights(weights_path)

    model.add(Dense(2, activation = 'softmax'))

    return model


def dense():
    model = Sequential()
    model.add(Flatten(input_shape = (160, 120, 3)))
    model.add(Dense(5000, activation = "relu"))
    model.add(Dropout(0.1))
    model.add(Dense(500))
    model.add(Dropout(0.1))
    model.add(Dense(50))
    model.add(Dropout(0.1))
    model.add(Dense(2, activation = "softmax"))

    return model

#### TRAIN
x_train, y_train, x_test = read_data()

# normalize
#x_train = x_train.astype('float32') / 255.
#x_test = x_test.astype('float32') / 255.

y_train = to_categorical(y_train, num_classes = 2)
#y_test = to_categorical(y_test, num_classes = 10)

print(x_train.shape)
print(y_train.shape)

model = use_base_model(ka.InceptionResNetV2)

# sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

model.compile('Nadam', "categorical_crossentropy")
model.fit(
    x_train, y_train,
    batch_size = 8,
    epochs = 4,
    shuffle = True
)

#preds = model.predict(x_train)
#print(np.argmax(preds, axis = 1))

preds = model.predict(x_test)
write_predictions(preds)

# InceptionV3 + 1024 + 128 + 2, adam, 4 epochs, CPU, loss = 0.2750
# VGG16 + 1024 + 128 + 2, adam, 4 epochs, CPU, loss = 0.0435
# XCeption + 1024 + 128 + 2, adam, 4 epochs, CPU, loss = 0.1602
# InceptionResNetV2 + 1024 + 128 + 2, adam, 4 epochs, CPU, loss = 0.1539
