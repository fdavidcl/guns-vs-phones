#!/usr/bin/env python

import numpy as np
from PIL import Image
import glob, os

from keras.preprocessing import image

def get_imgs(img_list):
    img = [image.load_img(filename, target_size = (224, 224), interpolation = "bicubic") for filename in img_list]
    print(len(img))
    npi = [np.array(i) for i in img]
    return np.stack(npi, axis = 0)    
        

def read_data():
    train_x_pistol = get_imgs(glob.glob("Train/Pistol/*"))
    train_x_phone = get_imgs(glob.glob("Train/Smartphone/*"))

    train_x = np.concatenate((train_x_pistol, train_x_phone), axis = 0)
    train_y = np.concatenate((np.zeros(train_x_pistol.shape[0]), np.ones(train_x_phone.shape[0])))
    
    test_x = get_imgs(glob.glob("Test/*"))

    return (train_x, train_y, test_x)

