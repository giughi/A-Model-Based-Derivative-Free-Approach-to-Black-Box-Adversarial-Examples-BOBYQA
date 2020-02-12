import tensorflow as tf
import numpy as np
import os
import pickle
import gzip
import urllib.request

from keras.engine.training import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Add, BatchNormalization,  Input
from keras.utils import np_utils
from keras.models import load_model
from keras import backend as K, regularizers

K.set_learning_phase(False) 
#set learning phase
# def extract_data(filename, num_images):
#     with gzip.open(filename) as bytestream:
#         bytestream.read(16)
#         buf = bytestream.read(num_images*28*28)
#         data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
#         data = (data / 255) - 0.5
#         data = data.reshape(num_images, 28, 28, 1)
#         return data

# def extract_labels(filename, num_images):
#     with gzip.open(filename) as bytestream:
#         bytestream.read(8)
#         buf = bytestream.read(1 * num_images)
#         labels = np.frombuffer(buf, dtype=np.uint8)
#     return (np.arange(10) == labels[:, None]).astype(np.float32)

# class MNIST:
#     def __init__(self):
#         if not os.path.exists("data"):
#             os.mkdir("data")
#             files = ["train-images-idx3-ubyte.gz",
#                      "t10k-images-idx3-ubyte.gz",
#                      "train-labels-idx1-ubyte.gz",
#                      "t10k-labels-idx1-ubyte.gz"]
#             for name in files:

#                 urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/' + name, "data/"+name)

#         train_data = extract_data("data/train-images-idx3-ubyte.gz", 60000)
#         train_labels = extract_labels("data/train-labels-idx1-ubyte.gz", 60000)
#         self.test_data = extract_data("data/t10k-images-idx3-ubyte.gz", 10000)
#         self.test_labels = extract_labels("data/t10k-labels-idx1-ubyte.gz", 10000)
        
#         VALIDATION_SIZE = 5000
        
#         self.validation_data = train_data[:VALIDATION_SIZE, :, :, :]
#         self.validation_labels = train_labels[:VALIDATION_SIZE]
#         self.train_data = train_data[VALIDATION_SIZE:, :, :, :]
#         self.train_labels = train_labels[VALIDATION_SIZE:]


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
        return self.model(data)