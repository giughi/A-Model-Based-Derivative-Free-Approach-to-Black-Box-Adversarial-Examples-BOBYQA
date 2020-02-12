
# coding: utf-8

# ## Image Classification using Convolutional Neural Networks
#
# In this example we will use Keras to build a deep Convolutional Neural Network (CNN) and train it on the STL-10 datase
#
# After 100 training epochs, this network scores ~71% top-1 accuracy on the dataset's test-set.
#
#

# ### Preparations
#
# First, lets import all the modules we use below.

# In[10]:


import os
# import urllib.request as urllib
import tarfile
import sys
sys.path.append("./")


from Setups.Data_and_Model.setup_STL10_2 import STL10Model

import keras
# import matplotlib.pyplot as plt
import numpy as np
import pickle
from keras import backend as K, regularizers
from keras.callbacks import LearningRateScheduler
from keras.engine.training import Model
from keras.layers import Add, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, Activation, Input
from keras.preprocessing.image import ImageDataGenerator



import tensorflow as tf
from Attack_Code.BOBYQA.BOBYQA_Attack_rand_multi_STL_BOUNDS import BlackBox_BOBYQA
from Attack_Code.GenAttack.genattack_tf_STL import GenAttack2
# Lets define a few constants we will later use

# the dimensions of each image in the STL-10 dataset (96x96x3).
HEIGHT, WIDTH, DEPTH = 96, 96, 3

# number of classes in the STL-10 dataset.
N_CLASSES = 10

# size of a single image in bytes
SIZE = HEIGHT * WIDTH * DEPTH

# path to the directory with the data
dir_path = os.path.dirname(os.path.realpath(__file__))
main_dir = os.path.abspath(os.path.join(dir_path, os.pardir))
results_dir = main_dir + '/Results/STL10/'

DATA_DIR = main_dir + '/Data/STL10'

# url of the binary data
DATA_URL = 'http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz'

# path to the binary train file with image data
TRAIN_DATA_PATH = DATA_DIR + '/train_X.bin'

# path to the binary test file with image data
TEST_DATA_PATH = DATA_DIR + '/test_X.bin'

# path to the binary train file with labels
TRAIN_LABELS_PATH = DATA_DIR + '/train_y.bin'

# path to the binary test file with labels
TEST_LABELS_PATH = DATA_DIR + '/test_y.bin'

# path to class names file
CLASS_NAMES_PATH = DATA_DIR + '/class_names.txt'


# Next, we need to download and load the SLT-10 dataset. Lets write a few helper functions to do that.
#

def read_labels(path_to_labels):
    with open(path_to_labels, 'rb') as f:
        labels = np.fromfile(f, dtype=np.uint8)
        return labels


def read_all_images(path_to_data):
    with open(path_to_data, 'rb') as f:
        # read whole file in uint8 chunks
        everything = np.fromfile(f, dtype=np.uint8)

        # We force the data into 3x96x96 chunks, since the
        # images are stored in "column-major order", meaning
        # that "the first 96*96 values are the red channel,
        # the next 96*96 are green, and the last are blue."
        # The -1 is since the size of the pictures depends
        # on the input file, and this way numpy determines
        # the size on its own.

        images = np.reshape(everything, (-1, DEPTH, WIDTH, HEIGHT))

        # Now transpose the images into a standard image format
        # readable by, for example, matplotlib.imshow
        # You might want to comment this line or reverse the shuffle
        # if you will use a learning algorithm like CNN, since they like
        # their channels separated.
        images = np.transpose(images, (0, 3, 2, 1))
        return images


def download_and_extract():
    # if the dataset already exists locally, no need to download it again.
    if all((
        os.path.exists(TRAIN_DATA_PATH),
        os.path.exists(TRAIN_LABELS_PATH),
        os.path.exists(TEST_DATA_PATH),
        os.path.exists(TEST_LABELS_PATH),
    )):
        return

    dest_directory = DATA_DIR
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)

    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\rDownloading %s %.2f%%' % (filename,
                                                          float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.urlretrieve(DATA_URL, filepath, reporthook=_progress)
        print('Downloaded', filename)
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def load_dataset():
    # download the extract the dataset.
    download_and_extract()

    # load the train and test data and labels.
    x_train = read_all_images(TRAIN_DATA_PATH)
    y_train = read_labels(TRAIN_LABELS_PATH)
    x_test = read_all_images(TEST_DATA_PATH)
    y_test = read_labels(TEST_LABELS_PATH)

    # convert all images to floats in the range [0, 1]
    x_train = x_train.astype('float32')
    x_train = (x_train - 127.5) / 127.5
    x_test = x_test.astype('float32')
    x_test = (x_test - 127.5) / 127.5

    # convert the labels to be zero based.
    y_train -= 1
    y_test -= 1

    # convert labels to hot-one vectors.
    y_train = keras.utils.to_categorical(y_train, N_CLASSES)
    y_test = keras.utils.to_categorical(y_test, N_CLASSES)

    return (x_train, y_train), (x_test, y_test)


# We can now load the dataset.
#

(x_train, y_train), (x_test, y_test) = load_dataset()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


# Let's now implement the session on teh model

nimage = 200 # len(x_test)
L_inf_classes = [0.20]
# print('Number of images', nimage)
# print(y_test)
successes = 0
total = 0
legend = []
with tf.Session() as sess:
    for L_inf in L_inf_classes:
        model = STL10Model(main_dir + "/Models/STL_riprova", sess)
        # attack = BlackBox_BOBYQA(sess, model, batch_size=50, max_iterations=100, print_every=1,
        #                          early_stop_iters=100, confidence=0, targeted=True, use_log=False,
        #                          use_tanh=False, use_resize=True, L_inf=L_inf, rank=3, ordered_domain=True,
        #                          image_distribution=True, mixed_distributions=False, max_eval=3000)
        k =0
        img = x_test[k, np.newaxis]

        img2 = x_test[k+10, np.newaxis]
        prediction2 = np.argmax(model.predict(img2)[0])

        prediction = np.argmax(model.predict(img)[0])

        print('Prediction1',prediction)
        print('Prediction2',prediction2)

        imgs = np.array([img[0], img2[0]])
        predictions = model.predict(imgs)
        print('Both Predictions',predictions)


        attack = GenAttack2(model=model,
                            pop_size=6,
                            mutation_rate=0.005,
                            eps=L_inf,
                            max_steps=int(3000/5),
                            alpha=0.2,
                            resize_dim=96,
                            adaptive=True)
        for k in range(100,nimage):
            img = x_test[k, np.newaxis]

            img2 = x_test[k+1, np.newaxis]
            prediction2 = np.argmax(model.model.predict(img2)[0])

            prediction = np.argmax(model.model.predict(img)[0])

            print('Prediction1',prediction)
            print('Prediction2',prediction2)

            imgs = np.array([img[0], img2[0]])
            predictions = np.argmax(model.model.predict(imgs))
            print('Both Predictions',predictions)

            real_class = np.argmax(y_test[k])
            if prediction != real_class:
                continue
            targets = []
            for i in range(10):
                if i != prediction:
                    targets.append(i)
            for target in targets:
                total +=1
                target_input = np.eye(10)[target, :]
                adv, eval_costs, _ = attack.attack_batch(sess, img[0], target)
                if len(adv.shape) == 3:
                    adv = adv.reshape((1,) + adv.shape)

                adversarial_predict = np.argmax(model.model.predict(adv))
                if adversarial_predict == target:
                    successes += 1
                legend.append([eval_costs, real_class, target, adversarial_predict])

                print("[STATS][L1] total = {}, L_inf = {}, prev_class = {}, target = {},  new_class = {}, success = {}".
                      format(target + k*10, L_inf, prediction, target, adversarial_predict, successes/total))
                sys.stdout.flush()

                with open(results_dir+'simple_attack_L_inf_GEN_'+str(L_inf/2)+'_max_eval_'+ str(3000) +'_2.txt', "wb") as fp:
                    pickle.dump(legend, fp)
