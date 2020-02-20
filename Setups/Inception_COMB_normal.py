import argparse
import inspect
import json
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
from PIL import Image
import sys
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

""" The collection of all attack classes """
from Attack_Code.Combinatorial.attacks.parsimonious_attack_function import ParsimoniousAttack_function

#   print('attack: ', attack)
#   print('sys.modules', sys.modules[__name__])

""" Arguments """
dir_path = os.path.dirname(os.path.realpath(__file__))
main_dir = os.path.abspath(os.path.join(dir_path, os.pardir))

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
flags.DEFINE_integer('max_internal_queries', default=1000, help='Maximum number of queries we are allowed to do before resetting the model')
flags.DEFINE_integer('jumping', default=0, help='amount of images to jump')
# flags.DEFINE_integer('target', default=704, help='Target that we want to attack')
flags.DEFINE_string('mod_sav', default='', help='modification on the saving name')
flags.DEFINE_bool('print_image', False, 'Bool on targeted attack.')
FLAGS = flags.FLAGS


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


    saving_dir = main_dir+'/Results/Imagenet/COMBI_'+str(FLAGS.epsilon)+'.txt'
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
            
            # cuda.select_device(0)
            # cuda.close()
            # # cuda.profile_stop()
            time_attack = 0
            tf.reset_default_graph()
            # tf.set
            K.clear_session()
            tf.GraphDef().Clear()

            print('=======> Iter ', count, 'L_inf', FLAGS.epsilon)
            count += 1

            session_conf = tf.ConfigProto(intra_op_parallelism_threads=10,
                                          inter_op_parallelism_threads=10)

            with tf.Session(config=session_conf) as sess:

                
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
                    # adv_image = image + cv2.resize(noise[0, ...], (self.width, self.height), interpolation=cv2.INTER_NEAREST)
                    resized_img = tf.image.resize_images(np.array([noise[0,...]]), [width, height],
                                                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=True).eval()
                    adv_image = image + resized_img
                    if width != 96:
                        adv_image = np.clip(adv_image, -0.5, 0.5)
                    else:
                        adv_image = np.clip(adv_image, -1, 1)
                    return adv_image


                print('Inside session')
                tf.set_random_seed(FLAGS.seed)
                model = InceptionModel(sess, use_log=True)
                test_in_2 = tf.placeholder(tf.float32, (None, 299, 299, 3), 'x')
                logit_pred = model.predict(test_in_2)
                test_pred_2 = tf.argmax(model.predict(test_in_2), axis=1)
                # Create attack
                # attack_class = getattr(sys.modules[__name__], FLAGS.attack)
                # attack = attack_class(model, FLAGS)
                
                if first_iter:
                    orig_pred = sess.run(test_pred_2, feed_dict={test_in_2: [input_img0]})[0]
                    logit_pr = sess.run(logit_pred,  feed_dict={test_in_2: [input_img0]})[0]
                    if FLAGS.verbose:
                        print('Real = {}, Predicted = {}, Target = {}'.format(
                            real_label, orig_pred, target_class))

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

                y_input = tf.placeholder(dtype=tf.int32, shape=[None])
                test_in = tf.placeholder(tf.float32, (None, FLAGS.dim_image, FLAGS.dim_image, FLAGS.num_channels), 'x')
                probs = model.predict(test_in)
                preds = tf.argmax(probs, axis=1)
                logits = tf.math.log(probs)

                batch_num = tf.range(0, limit=tf.shape(probs)[0])
                indices = tf.stack([batch_num, y_input], axis=1)
                # indices_sum = tf.stack([batch_num, self.y_input], axis=1)
                ground_truth_probs = tf.gather_nd(params=probs, indices=indices)
                # ground_truth_probs_sum = tf.gather_nd(params=probs, indices=indices)
                top_2 = tf.nn.top_k(probs, k=2)
                max_indices = tf.where(tf.equal(top_2.indices[:, 0], y_input), top_2.indices[:, 1], top_2.indices[:, 0])
                max_indices = tf.stack([batch_num, max_indices], axis=1)
                max_probs = tf.gather_nd(params=probs, indices=max_indices)

                if FLAGS.no_hier:
                    losses = -(ground_truth_probs + 1e-10) + (max_probs + 1e-10)
                else:
                    losses = -tf.log(ground_truth_probs + 1e-10) + tf.log(tf.reduce_sum(probs, 1) 
                                                                        - ground_truth_probs + 1e-10)

                

                adv_img, num_queries, attack_completed, values, noise = ParsimoniousAttack_function(model,
                    input_img, target_class, sess, values, FLAGS.max_internal_queries, noise, 
                    y_input, test_in, losses, preds, _perturb_image, FLAGS)
                end_time = time.time()
                
                assert np.amax(np.abs(adv_img-input_img0)) <= FLAGS.epsilon+1e-3
                assert np.amax(adv_img) <= +0.5+1e-3
                assert np.amin(adv_img) >= -0.5-1e-3

                time_attack += (end_time-start_time)   
                
                if attack_completed:
                    full_adversarial = adv_img
                    final_pred = sess.run(test_pred_2, feed_dict={test_in_2: full_adversarial})[0]
                    print('Predicted',final_pred)
                    print('Target', target_class[0])
                    if final_pred == target_class[0]:
                        success_count += 1
                        print('--- SUCCEEEED ----')
                        # logger.add_result(ii, input_img, adv_img, real_label,
                        #     target_label, query_count, attack_time, margin_log)
                
                sess.close()
                del model
                gc.collect()

        if not right_classification:
            continue

        frob_norm = np.linalg.norm(adv_img-input_img0)

        attacks.append([num_queries, real_label, target_class, values['loss'], frob_norm])
        with open(saving_dir, "wb") as fp:
                    pickle.dump(attacks, fp)
            # logger.close(num_attempts=total_count)
        print('Number of success = {} / {}.'.format(success_count, total_count))