import tensorflow as tf
import numpy as np
import os
import pickle
import gzip
# import urllib.request

from keras.engine.training import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Add, BatchNormalization,  Input
from keras.utils import np_utils
from keras.models import load_model
from keras import backend as K, regularizers

K.set_learning_phase(False) 


class STL10Model:
    def __init__(self, restore=None, session=None, use_log=False):
        n_conv_blocks = 5  # number of convolution blocks to have in our model.
        n_filters = 32  # number of filters to use in the first convolution block.
        l2_reg = regularizers.l2(2e-4)  # weight to use for L2 weight decay. 
        activation = 'relu'

        self.num_channels = 3
        self.image_size = 96
        self.num_labels = 10

        if restore:
            model = load_model(restore)
        self.model = model

    def predict(self, data):
        return self.model.predict(data)