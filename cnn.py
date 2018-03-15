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
from keras.optimizers import SGD, RMSprop
from keras.models import Sequential, Model
from keras.utils.np_utils import to_categorical
from keras.applications.imagenet_utils import preprocess_input
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
    #x = Flatten()(x)
    x = Dense(512, activation = 'relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation = 'relu')(x)
    x = Dropout(0.2)(x)
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

# preprocess
# what this does: https://github.com/keras-team/keras/blob/master/keras/applications/imagenet_utils.py#L24
x_train = preprocess_input(np.float64(x_train), mode = 'caffe')
x_test = preprocess_input(np.float64(x_test), mode = 'caffe')

y_train = to_categorical(y_train, num_classes = 2)

print(x_train.shape)
print(y_train.shape)

model = use_base_model(VGG16)

# Alternative rmsprop:
rms = RMSprop(lr=0.002, rho=0.9, epsilon=None, decay=0.001)

model.compile("rmsprop", "categorical_crossentropy")
model.fit(
    x_train, y_train,
    batch_size = 16,
    epochs = 10,
    shuffle = True
)

preds = model.predict(x_test)
write_predictions(preds)

# InceptionV3 + 1024 + 128 + 2, adam, 4 epochs, CPU, loss = 0.2750
# VGG16 + 1024 + 128 + 2, adam, 4 epochs, CPU, loss = 0.0435
# VGG16 + 1024 + 128 + 2 (dropout 0.5), rmsprop, 4 epochs, GPU, loss = 0.1407
# VGG16 + 1024 + 128 + 2 (dropout 0.2), rmsprop, 4 epochs, GPU, loss = 0.0436
# VGG16 + 1024 + 128 + 2 (dropout 0.2), nadam, 4 epochs, GPU, loss = 8.5050
# VGG16 + 1024 + 128 + 2 (dropout 0.2), rmsprop lr=0.002 decay=0.001, 8..10 epochs, GPU, loss = 0.0218
# VGG16 + 512 + 64 + 2 (dropout 0.2), rmsprop lr=0.002 decay=0.001, 9..10 epochs, GPU, loss = 1e-7
# VGG16 + 512 + 64 + 2 (dropout 0.2), rmsprop lr=0.001 decay=0, 10 epochs, GPU, loss = 3e-7
# pre[caffe] VGG16 + 512 + 64 + 2 (dropout 0.2), rmsprop lr=0.001 decay=0, 10 epochs, GPU, loss = 3.8e-7
# pre[bicubic,220x220,caffe] VGG16 + 512 + 64 + 2 (dropout 0.2), rmsprop lr=0.001 decay=0, 10 epochs, GPU, loss = 1e-7
# pre[caffe] VGG16 + 512 + 64 + 2 (dropout 0.2), rmsprop lr=0.002 decay=0.001, 10 epochs, GPU, loss = 0.0218
# pre[tf] VGG16 + 512 + 64 + 2 (dropout 0.2), rmsprop lr=0.001 decay=0, 10 epochs, GPU, loss = 0.0042
# pre[tf] VGG16 + 512 + 64 + 2 (dropout 0.2), rmsprop lr=0.002 decay=0.001, 10 epochs, GPU, loss = 6.8e-5
# pre[tprch] VGG16 + 512 + 64 + 2 (dropout 0.2), rmsprop lr=0.001 decay=0, 10 epochs, GPU, loss = 1.5e-6
# pre[tprch] VGG16 + 512 + 64 + 2 (dropout 0.2), rmsprop lr=0.002 decay=0.001, 10 epochs, GPU, loss = 6.1e-6
# XCeption + 1024 + 128 + 2, adam, 4 epochs, CPU, loss = 0.1602
# InceptionResNetV2 + 1024 + 128 + 2, adam, 4 epochs, CPU, loss = 0.1539
