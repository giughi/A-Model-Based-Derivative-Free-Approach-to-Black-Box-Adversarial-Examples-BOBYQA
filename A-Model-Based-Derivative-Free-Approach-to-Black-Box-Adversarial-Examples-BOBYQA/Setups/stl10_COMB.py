
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
import urllib.request as urllib
import tarfile
import sys
sys.path.append("./")
import inspect

import pickle

from Setups.Data_and_Model.setup_STL10 import STL10Model
from Attack_Code.Combinatorial import attacks
from Attack_Code.Combinatorial.tools.imagenet_labels import *


import keras
# import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K, regularizers
from keras.callbacks import LearningRateScheduler
from keras.engine.training import Model
from keras.layers import Add, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, Activation, Input
from keras.preprocessing.image import ImageDataGenerator


import tensorflow as tf
from Attack_Code.BOBYQA.BOBYQA_Attack_rand_multi_STL_BOUNDS import BlackBox_BOBYQA

# uploading attacks
ATTACK_CLASSES = [x for x in attacks.__dict__.values() if inspect.isclass(x)]
for attack in ATTACK_CLASSES:
  setattr(sys.modules[__name__], attack.__name__, attack)
  print('attack: ', attack)
  print('sys.modules', sys.modules[__name__])


# path to the directory with the data
dir_path = os.path.dirname(os.path.realpath(__file__))
main_dir = os.path.abspath(os.path.join(dir_path, os.pardir))
results_dir = main_dir + '/Results/STL10/'
DATA_DIR = main_dir + '/Data/STL10'

# Lets define a few constants we will later use
flags = tf.app.flags
flags.DEFINE_integer('max_evals', 3000, 'Maximum number of function evaluations.')
flags.DEFINE_integer('print_every', 10, 'Every iterations the attack function has to print out.')
flags.DEFINE_integer('seed', 1216, 'random seed')
flags.DEFINE_string('description', '', 'Description for how to save the results')

# flags for combinatorial attack
flags.DEFINE_string('asset_dir', default=main_dir+'/Attack_Code/Combinatorial/assets', help='directory assets')
flags.DEFINE_integer('batch', 50, help='Dimension of the sampling domain.')
flags.DEFINE_integer('img_index_start', default=0, help='first image')

flags.DEFINE_float('epsilon', 0.10, help='maximum L_inf distance threshold') #######
flags.DEFINE_integer('max_steps', 1000, help='Maximum number of iterations')
flags.DEFINE_integer('resize_dim', 96, help='Reduced dimension for dimensionality reduction') #######
# flags.DEFINE_string('model', default='inception', help='model name') ######
flags.DEFINE_string('loss_func', default='cw', help='The type of loss function') ######
flags.DEFINE_integer('target', 704, help='target class. if not provided will be random')
flags.DEFINE_string('attack', default='ParsimoniousAttack_2', help='The type of attack')
flags.DEFINE_integer('max_queries', default=4000, help='The query limit') ##########
flags.DEFINE_bool('targeted', default=True, help='bool on targeted')
flags.DEFINE_integer('max_iters', 1, help='maximum iterations') ##########
flags.DEFINE_integer('block_size', default=32, help='blck size') ##########
flags.DEFINE_integer('batch_size', default=64, help='batch size') ##########
flags.DEFINE_bool('no_hier', default=False, help='bool on hierarchical attack') #########
flags.DEFINE_string('input_dir', default='', help='Directory from whih to take the inputs') ########
flags.DEFINE_integer('dim_image', default=96, help='Dimension of the image that we feed as an input')
flags.DEFINE_integer('num_channels', default=3, help='Channels of the image that we feed as an input')

FLAGS = flags.FLAGS
# the dimensions of each image in the STL-10 dataset (96x96x3).
HEIGHT, WIDTH, DEPTH = 96, 96, 3

# number of classes in the STL-10 dataset.
N_CLASSES = 10

# size of a single image in bytes
SIZE = HEIGHT * WIDTH * DEPTH


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


# Notice that we have 5000 images in our training set, and 8000 images in out test set.
# Each image is of dimension 96x96 and has three color channels (RGB).
# Let's take a look at how these images look like:
#

def plot_images(images, n_images):
    _, h, w, d = images.shape
    # create an array that will store the images to plot.
    canvas = np.empty((h * n_images, w * n_images, d), dtype='uint8')

    for i in range(n_images):
        img_column = images[i * n_images:(i + 1) * n_images]
        for j in range(n_images):
            if j >= img_column.shape[0]:
                break

            # transform images to the range [0, 255]
            img = img_column[j]
            img = ((img * 127.5) + 127.5).clip(0, 255).astype('uint8')
            canvas[i * h:(i + 1) * h, j * w:(j + 1) * w] = img

    plt.figure(figsize=(2 * n_images, 2 * n_images))
    plt.axis('off')
    cmap = 'gray' if d == 1 else None
    plt.imshow(canvas.squeeze(), origin="upper", cmap=cmap)
    plt.show()


# plot_images(x_train, 10)

# Let's now implement the session on teh model

nimage = 1000 #len(x_test)
L_inf_classes = [0.2]
# print('Number of images', nimage)
# print(y_test)
successes = 0
total = 0

legend = []


for j in range(3):
    tf.reset_default_graph()
    with tf.Session() as sess:
        for L_inf in L_inf_classes:
            model = STL10Model(main_dir + "/Models/STL_riprova", sess)
            FLAGS.epsilon = L_inf
            # FLAGS.max_iters = 3000
            attack_class = getattr(sys.modules[__name__], FLAGS.attack)
            print(model)
            attack_combi = attack_class(model, FLAGS)

            for k in range(nimage):
                img = x_test[k, np.newaxis]
                            
                prediction = np.argmax(model.model.predict(img)[0])
                
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
                    # print('THE IMAGE HAS MAXIMUM', np.max(img),' AND MINIMUM ', np.min(img))
                    adv, eval_costs, succ_combi = attack_combi.perturb(img[0],np.array([np.argmax(target_input)]), 0, sess)
                    if succ_combi:
                        print('Succesful attack with ', eval_costs, 'queries')

                    if len(adv.shape) == 3:
                        adv = adv.reshape((1,) + adv.shape)

                    adversarial_predict = np.argmax(model.model.predict(adv))
                    if adversarial_predict == target:
                        successes += 1
                        # print(' With a norm of L-Inf',FLAGS.epsilon, 'We obatin and error of max',
                        #       np.max(adv-img),' and minimum', np.min(adv-img))
                        
                    legend.append([eval_costs, real_class, target])

                    print("[STATS][L1] total = {}, L_inf = {}, prev_class = {}, target = {},  new_class = {}, success = {}".
                        format(target + k*10, L_inf, prediction, target, adversarial_predict, successes/total))
                    sys.stdout.flush()
                    
                    with open(results_dir+'combi_simple_attack_L_inf_'+str(L_inf/2)+'_max_eval_'+ str(3000) +'_2.txt', "wb") as fp:
                        pickle.dump(legend, fp)