"""
python Setups/main_COMBINATORIAL_2.py --input_dir=Data/ImageNet/images --test_size=1 \
    --eps=0.05 \
    --max_steps=500000 --output_dir=attack_outputs  \
    --target=704

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import inspect
import json
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
from PIL import Image
import sys
import cv2
sys.path.append("./")
import random
import tensorflow as tf
import time
import gc
from keras import backend as K, regularizers
from numba import cuda

from Attack_Code.Combinatorial import attacks
from Attack_Code.Combinatorial.tools.imagenet_labels import *
from Attack_Code.Combinatorial.tools.utils import *
from Setups.Data_and_Model.setup_inception_2 import ImageNet, InceptionModel
from Attack_Code.GenAttack import utils

from tensorflow.contrib.slim.nets import inception
slim = tf.contrib.slim
from tensorflow.python.framework import ops
from tensorflow.python.training import monitored_session
from tensorflow.python.training import saver as tf_saver


""" The collection of all attack classes """
from Attack_Code.Combinatorial.attacks.parsimonious_attack_function_adv_inception import ParsimoniousAttack_function_adv_inception

#   print('attack: ', attack)
#   print('sys.modules', sys.modules[__name__])

""" Arguments """
dir_path = os.path.dirname(os.path.realpath(__file__))
main_dir = os.path.abspath(os.path.join(dir_path, os.pardir))
checkpoint_path_ = main_dir + '/Models/adv_inception_v3.ckpt'

IMAGENET_PATH =  main_dir + '/Data/ImageNet/images'
NUM_LABELS = 1000

parser = argparse.ArgumentParser()

# Directory

flags = tf.app.flags
flags.DEFINE_string('asset_dir', default=main_dir+'/Attack_Code/Combinatorial/assets', help='directory assets')
flags.DEFINE_string('save_dir', default=main_dir+'/Results/Imagenet', help='directory to save results')
flags.DEFINE_string('save_img', default=main_dir+'/Results/Imagenet/Images', help='store_true')
flags.DEFINE_integer('sample_size', 1000, help='Number of test images.')
flags.DEFINE_bool('verbose', True, help='Print logs.')
flags.DEFINE_integer('batch', 50, help='Dimension of the sampling domain.')
flags.DEFINE_integer('img_index_start', default=0, help='first image')

flags.DEFINE_float('epsilon', 0.10, help='maximum L_inf distance threshold')
flags.DEFINE_integer('max_steps', 1000, help='Maximum number of iterations')
flags.DEFINE_integer('resize_dim', 96, help='Reduced dimension for dimensionality reduction')
flags.DEFINE_string('model', default='inception', help='model name')
flags.DEFINE_string('loss_func', default='cw', help='The type of loss function')
flags.DEFINE_integer('seed', 1216, help='random seed')
flags.DEFINE_integer('target', None, help='target class. if not provided will be random')
flags.DEFINE_string('attack', default='ParsimoniousAttack_restarts', help='The type of attack')
flags.DEFINE_integer('max_queries', default=15000, help='The query limit')
flags.DEFINE_bool('targeted', default=True, help='bool on targeted')
flags.DEFINE_integer('max_iters', 1, help='maximum iterations')
flags.DEFINE_integer('block_size', default=32, help='blck size')
flags.DEFINE_integer('batch_size', default=64, help='batch size')
flags.DEFINE_bool('no_hier', default=False, help='bool on hierarchical attack')
flags.DEFINE_string('input_dir', default='', help='Directory from whih to take the inputs')
flags.DEFINE_integer('num_channels', default=3, help='Channels of the image that we feed as an input')
flags.DEFINE_integer('dim_image', default=299, help='Dimension of the image that we feed as an input')
flags.DEFINE_integer('max_internal_queries', default=17000, help='Maximum number of queries we are allowed to do before resetting the model')
flags.DEFINE_integer('jumping', default=0, help='amount of images to jump')
# flags.DEFINE_integer('target', default=704, help='Target that we want to attack')
flags.DEFINE_string('mod_sav', default='', help='modification on the saving name')
flags.DEFINE_bool('print_image', False, 'Bool on targeted attack.')
flags.DEFINE_bool('worst_attack', False, 'Bool on if the image to be attacked is the most difficult')


FLAGS = flags.FLAGS

def create_model(x, reuse=None):
    """Create model graph.
    Args:
    x: input images
    reuse: reuse parameter which will be passed to underlying variable scopes.
      Should be None first call and True every subsequent call.
    Returns:
    (logits, end_points) - tuple of model logits and enpoints
    Raises:
    ValueError: if model type specified by --model_name flag is invalid.
    """
    with slim.arg_scope(inception.inception_v3_arg_scope()):
          return inception.inception_v3(
              x, num_classes=1001, is_training=False, reuse=reuse)


class Adv_Inception():
    def __init__(self, sess):
        self.sess = sess

    def predict(self, img):
        
        return self.sess.run(logits, feed_dict={tf_image:2*img})

if __name__ == '__main__':
    # Set verbosity

    attacks_done = []

    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    dataset = ImageNet(FLAGS.input_dir)
    inputs, targets, reals, paths = utils.generate_data(dataset, FLAGS.sample_size)
    # print(inputs[0])
    #  Create a session

    num_valid_images = 1000  # len(inputs)
    total_count = 0  # Total number of images attempted
    success_count = 0
    attacks = []
#         logger = utils.ResultLogger(FLAGS.output_dir, FLAGS.flag_values_dict())
    if FLAGS.print_image:
        saving_dir = main_dir+'/Results/Imagenet/Adv_Images/ADV_COMBI_'+str(FLAGS.epsilon)+'_targeted_'+str(FLAGS.targeted)
        num_valid_images = 5
    else:
        if FLAGS.worst_attack:
            saving_dir = main_dir+'/Results/Imagenet/ADV_COMBI_'+str(FLAGS.epsilon)+'_function_jumping_'+str(FLAGS.jumping)+'_WORST_ATTACK.txt'
            print(saving_dir)
        else:
            saving_dir = main_dir+'/Results/Imagenet/ADV_COMBI_'+str(FLAGS.epsilon)+'_function_jumping_'+str(FLAGS.jumping)+'.txt'
        already_done = 0
        if os.path.exists(saving_dir):
            if os.path.getsize(saving_dir)>0:
                with open(saving_dir, "rb") as fp:
                    attacks = pickle.load(fp)
                    already_done = len(attacks)
        if FLAGS.jumping>0:
            already_done += FLAGS.jumping
        print('Already done ==============',already_done)
    for ii in range(already_done, num_valid_images):
        
        attack_completed = False
        
        input_img = inputs[ii]
        input_img0 = input_img.copy()
        input_img_path = paths[ii]
        # print(type(input_img))

        target_class = np.array([np.argmax(targets[ii])])
        print('==> Target Class',target_class)
        real_label = reals[ii]

        steps_done = -1
        perturbed_img = 0
        ord_domain = 0
        iteration_scale = 0
        query_count = 0

        count = 0

        values = None

        right_classification = True
        first_iter = True
        noise = None


        while not attack_completed:
            
            time_attack = 0
            tf.reset_default_graph()
            # tf.set
            K.clear_session()
            tf.GraphDef().Clear()

            print('=======> Iter ', count, 'L_inf', FLAGS.epsilon)
            count += 1

            session_conf = tf.ConfigProto(intra_op_parallelism_threads=5,
                                          inter_op_parallelism_threads=5)


            tf.logging.set_verbosity(0)#tf.logging.INFO)
            tf_global_step = tf.train.get_or_create_global_step()
            create_model(tf.placeholder(tf.float32, shape=[None,299,299,3]))
                
                
            tf_image = tf.placeholder(tf.float32, shape=[None,299,299,3])
            logits, _ = create_model(tf_image, reuse=True)
            variables_to_restore = slim.get_variables_to_restore()

            saver = tf_saver.Saver(variables_to_restore)

            graph = ops.get_default_graph()


            session_creator = monitored_session.ChiefSessionCreator(
                scaffold=monitored_session.Scaffold(
                    init_op=None, init_feed_dict=None, saver=saver),
                checkpoint_filename_with_path=checkpoint_path_,
                master='',
                config=session_conf)


            with monitored_session.MonitoredSession(
                session_creator=session_creator, hooks=None) as sess:
            #     if eval_ops is not None:
            #         while not session.should_stop():

                model = Adv_Inception(sess)
                
                logits_in1 = model.predict(np.array([inputs[0]]))
                
                logits_in = sess.run(logits, feed_dict={tf_image:np.array([inputs[0]]) * 2.0})
                        
                # print('1', logits_in1, '2', logits_in)

                
                def _perturb_image(width, height, image, noise):
                    """Given an image and a noise, generate a perturbed image.
                    First, resize the noise with the size of the image.
                    Then, add the resized noise to the image.

                    Args:
                        image: numpy array of size [1, 299, 299, 3], an original image
                        noise: numpy array of size [1, 256, 256, 3], a noise

                    Returns:
                        adv_iamge: numpy array of size [1, 299, 299, 3], an perturbed image
                    """
                    adv_image = image + cv2.resize(noise[0, ...], (width, height), interpolation=cv2.INTER_NEAREST)
                    # resized_img = tf.image.resize_images(np.array([noise[0,...]]), [width, height],
                    #                                     method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=True).eval()
                    # adv_image = image + resized_img
                    if width != 96:
                        adv_image = np.clip(adv_image, -0.5, 0.5)
                    else:
                        adv_image = np.clip(adv_image, -1, 1)
                    return adv_image


                print('Inside session')
                tf.set_random_seed(FLAGS.seed)
                if first_iter:
                    orig_pred = np.argmax(sess.run(logits, feed_dict={tf_image: 2*[input_img0]})[0])
                    if FLAGS.verbose:
                        print('Real = {}, Predicted = {}, Target = {}'.format(
                            real_label, orig_pred, target_class))
                    if FLAGS.worst_attack:
                        worst_class = np.argmin(logits_in)
                        if worst_class != np.argmin(logits_in1):
                            Warning('**************************THE MODEL IS BEHAVING WIERDLY')
                        target_class = [worst_class]
                        print('Because of Worst Case target is', worst_class)
                    if orig_pred != real_label:
                        if FLAGS.verbose:
                            print('\t Skipping incorrectly classified image.')
                        right_classification = False
                        break
                    total_count += 1
                    first_iter=False
                    orig_class = orig_pred

                start_time = time.time()
                print('Inside main_MBAE the target is', np.argmax(target_class))

                # def softmax(x):
                #     return np.exp(x)/sum(np.exp(x))

                def loss_func(img):
                    nn= len(img)
                    if nn==299:
                        img = [img]
                        nn = 1
                    lll = []
                    preds_l = []
                    for i in range(nn):
                        logits_ = model.predict([img[i]])
                        probs_ = softmax(logits_[0])
                        indices = target_class[0]
                        lll.append(-np.log(probs_[indices] + 1e-10) + np.log(np.sum(probs_) 
                                                                               - probs_[indices] + 1e-10))
                        preds_l.append(np.argmax(logits_[0]))
                    return lll, preds_l

                
                print('Loss', loss_func( [input_img0]))


                adv_img, num_queries, attack_completed, values, noise = ParsimoniousAttack_function_adv_inception(loss_func,
                    input_img, values, FLAGS.max_internal_queries, noise, _perturb_image, target_class[0], FLAGS)
                end_time = time.time()
                
                assert np.amax(np.abs(adv_img-input_img0)) <= FLAGS.epsilon+1e-3
                assert np.amax(adv_img) <= +0.5+1e-3
                assert np.amin(adv_img) >= -0.5-1e-3

                time_attack += (end_time-start_time)   
                
                if attack_completed:
                    full_adversarial = adv_img
                    final_pred = np.argmax(model.predict(np.array([full_adversarial]))[0])
                    print('Predicted',final_pred)
                    print('Target', target_class[0])
                    if final_pred == target_class[0]:
                        success_count += 1
                        print('--- SUCCEEEED ----')
                        # logger.add_result(ii, input_img, adv_img, real_label,
                        #     target_label, query_count, attack_time, margin_log)
                
                # sess.close()
                # gc.collect()

        if not right_classification:
            continue

        frob_norm = np.linalg.norm(adv_img-input_img0)
        if FLAGS.print_image:
            if final_pred == target_class[0]:
                to_save = [input_img0, adv_img]
            else:
                to_save = []
            with open(saving_dir+'_img_n_'+str(ii)+'.txt', "wb") as fp:
                        pickle.dump(to_save, fp)
        else:        
            attacks.append([num_queries, real_label, target_class, values['loss'], frob_norm])
            with open(saving_dir, "wb") as fp:
                        pickle.dump(attacks, fp)
            # logger.close(num_attempts=total_count)
        print('Number of success = {} / {}.'.format(success_count, total_count))