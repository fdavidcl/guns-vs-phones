#!/usr/bin/env python
import numpy as np
np.random.seed(1234) # for reproducibility

# https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
import os
import argparse
import logging

os.environ['PYTHONHASHSEED'] = '0'
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, GlobalAveragePooling2D
from keras.layers.convolutional import MaxPooling2D
from keras.optimizers import RMSprop
from keras.models import Sequential, Model
from keras.utils.np_utils import to_categorical
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.vgg16 import VGG16

from read_data import read_data
from write_predictions import write_predictions
from save_model import save_my_model
from save_model import load_my_model

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

def main():
    logger = logging.getLogger('cnn.py')
    logging.basicConfig()
    logger.setLevel(logging.INFO)
    parser = argparse.ArgumentParser(description='Train or predict if a image is a gun or phone')
    parser.add_argument('-p','--pretrained',
                        help='Use an existing trained model',
                        type=str,
                        nargs='+',
                        required=False)

    args = parser.parse_args()
    logger.info(str(args.pretrained))

    #### TRAIN
    x_train, y_train, x_test = read_data()
    x_test = preprocess_input(np.float64(x_test), mode = 'caffe')
    # preprocess
    # what this does: https://github.com/keras-team/keras/blob/master/keras/applications/imagenet_utils.py#L24
    x_train = preprocess_input(np.float64(x_train), mode = 'caffe')
    y_train = to_categorical(y_train, num_classes = 2)

    # Alternative rmsprop:
    rms = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0)

    if args.pretrained:
        logger.info('using existing model %s...', args.pretrained)
        model_name = args.pretrained[0]
        w = args.pretrained[1]
        model = load_my_model(model = model_name,
                              weights = w)
        model.compile(rms, "categorical_crossentropy")
    else:
        logger.info('Training a new model...')

        model = use_base_model(VGG16)
        model.compile("rmsprop", "categorical_crossentropy")
        model.fit(
            x_train, y_train,
            batch_size = 16,
            epochs = 10,
            shuffle = True
        )

    logger.info('Predicting...')
    preds = model.predict(x_test)
    logger.info('Writing predictions...')
    write_predictions(preds)

    # Save model and weights
    if not args.pretrained:
        save = raw_input("Save model? [y/n]: ")
        if save == 'y':
            name = raw_input('Name the model: ')
            save_my_model(model,
                          modelname = name,
                          w = name)

if __name__ == '__main__':
    main()

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
# pre[tf] VGG19 + 512 + 64 + 2 (dropout 0.2), rmsprop lr=0.002 decay=0.001, 10 epochs, GPU, loss = 0.0045
# pre[bicubic,224x224,tf] VGG16 + 512 + 64 + 2 (dropout 0.2), rmsprop lr=0.001 decay=0, 10 epochs, GPU, loss = loss: 0.0056 (3 missclass)\
# pre[torch] VGG16 + 512 + 64 + 2 (dropout 0.2), rmsprop lr=0.002 decay=0.001, 7 epochs, CPU, loss = 0.0055 (4 mal clasif)
# pre[torch] Xception + 512 + 64 + 2 (dropout 0.2), rmsprop lr=0.002 decay=0.001, 7 epochs, CPU, loss = 0.0359 (+20 mal clasif)

