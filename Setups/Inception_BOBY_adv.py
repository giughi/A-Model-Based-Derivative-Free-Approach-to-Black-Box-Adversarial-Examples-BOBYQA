"""
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import sys
sys.path.append("./")
import random
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_CONTROL_FLOW_V2']="1"

dir_path = os.path.dirname(os.path.realpath(__file__))
main_dir = os.path.abspath(os.path.join(dir_path, os.pardir))
checkpoint_path_ = main_dir + '/Models/adv_inception_v3.ckpt'

import tensorflow as tf   
from Attack_Code.BOBYQA.BOBYQA_Attack_Adversary import BlackBox_BOBYQA
from Setups.Data_and_Model.setup_inception_2 import ImageNet, InceptionModel
import pickle
from keras import backend as K, regularizers
import Attack_Code.GenAttack.utils as utils

from tensorflow.contrib.slim.nets import inception
slim = tf.contrib.slim
from tensorflow.python.framework import ops
from tensorflow.python.training import monitored_session
from tensorflow.python.training import saver as tf_saver

flags = tf.app.flags
flags.DEFINE_string('input_dir', '', 'Path for input images.')
flags.DEFINE_string('output_dir', 'output', 'Path to save results.')
flags.DEFINE_integer('test_size', 1000, 'Number of test images.')
flags.DEFINE_bool('verbose', True, 'Print logs.')
flags.DEFINE_integer('batch', 50, 'Dimension of the sampling domain.')
flags.DEFINE_integer('test_example', None, 'Test only one image')

flags.DEFINE_float('eps', 0.10, 'maximum L_inf distance threshold')
flags.DEFINE_integer('max_eval', 15000, 'Maximum number of iterations')
flags.DEFINE_integer('resize_dim', 96, 'Reduced dimension for dimensionality reduction')
flags.DEFINE_string('model', 'inception', 'model name')
flags.DEFINE_string('interp', 'over', 'kind of interpolation done on the data, either  <<over>> or <<linear>>')
flags.DEFINE_integer('seed', 1216, 'random seed')
flags.DEFINE_integer('target', None, 'target class. if not provided will be random')
flags.DEFINE_integer('jumping',0,'images to skip')
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

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

if __name__ == '__main__':

    attacks_done = []
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    dataset = ImageNet(FLAGS.input_dir)
    inputs, targets, reals, paths = utils.generate_data(dataset, FLAGS.test_size)
    
    num_valid_images = 1000  # len(inputs)
    total_count = 0  # Total number of images attempted
    success_count = 0
    attacks = []
#         logger = utils.ResultLogger(FLAGS.output_dir, FLAGS.flag_values_dict())
    saving_dir = main_dir+'/Results/Imagenet/ADV_BOBY_'+str(FLAGS.eps)+'.txt'
    print(saving_dir)
    saving_dir_full = main_dir+'/Results/Imagenet/ADV_BOBY_'+str(FLAGS.eps)+'.txt'
    already_done = 0
    if os.path.exists(saving_dir_full):
        if os.path.getsize(saving_dir_full)>0:
            with open(saving_dir_full, "rb") as fp:
                attacks = pickle.load(fp)
                already_done = len(attacks) + 1
    print('=====> Already done ', already_done)

    for ii in range(already_done+FLAGS.jumping, num_valid_images):
        
        attack_completed = False
        
        input_img = inputs[ii]
        img0 = input_img.copy()
        input_img_path = paths[ii]
        print(type(input_img))
        # np.save('Attack_Results/input_img', input_img)
        # print('saved_image')
        if FLAGS.target:
            target_label = np.eye(len(targets[ii]))[FLAGS.target + 1]
        else:
            target_label = targets[ii]
        real_label = reals[ii]

        steps_done = -1
        perturbed_img = 0
        ord_domain = 0
        iteration_scale = 0
        query_count = 0
        permutation = None
        right_classification = True

        first_iter = True

        while not attack_completed:

            time_attack = 0
            tf.reset_default_graph()
            # tf.set
            K.clear_session()
            tf.GraphDef().Clear()

            session_conf = tf.ConfigProto(intra_op_parallelism_threads=12,
                                          inter_op_parallelism_threads=12)

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

            # session_conf = tf.ConfigProto(
            #       intra_op_parallelism_threads=1,
            #       inter_op_parallelism_threads=1)
            # sess = tf.Session(config=session_conf)

            with monitored_session.MonitoredSession(
                session_creator=session_creator, hooks=None) as sess:
            #     if eval_ops is not None:
            #         while not session.should_stop():

                model = Adv_Inception(sess)
                
                logits_in1 = model.predict(np.array([inputs[0]]))
                
                logits_in = sess.run(logits, feed_dict={tf_image:np.array([inputs[0]]) * 2.0})
                        
                tf.set_random_seed(FLAGS.seed)

                if first_iter:
                    orig_pred = np.argmax(sess.run(logits, feed_dict={tf_image: 2*[img0]})[0])
                    if FLAGS.verbose:
                        print('Real = {}, Predicted = {}, Target = {}'.format(
                            real_label, orig_pred, np.argmax(target_label)))
                    if orig_pred != real_label:
                        if FLAGS.verbose:
                            print('\t Skipping incorrectly classified image.')
                        right_classification = False
                        break
                    total_count += 1
                    first_iter=False

                def loss_func(img, pert, only_loss=False):
                    nn= len(pert)
                    if nn==299:
                        img = [img]
                        nn = 1
                    lll = []
                    preds_l = []
                    distances = []
                    
                    for i in range(nn):
                        logits_ = model.predict([img + pert[i]])
                        probs_ = softmax(logits_[0])
                        indices = np.argmax(target_label)
                        lll.append(-np.log(probs_[indices] + 1e-10) + np.log(np.sum(probs_)- probs_[indices] + 1e-10))
                        preds_l.append(np.argmax(logits_[0]))
                        distances.append(np.max(probs_)-probs_[indices])
                    
                    if only_loss:
                        return lll
                    else:
                        return lll, logits_, distances

                
                attack = BlackBox_BOBYQA(loss_func, batch_size=FLAGS.batch , max_iterations=10000, print_every=1, 
                                        early_stop_iters=100, confidence=0, targeted=True, use_resize=True, 
                                        L_inf=FLAGS.eps, max_eval=FLAGS.max_eval,
                                        done_eval_costs=query_count, max_eval_internal=2000, 
                                        perturbed_img=perturbed_img, ord_domain=ord_domain, steps_done=steps_done,
                                        iteration_scale=iteration_scale,image0=img0, over=FLAGS.interp, 
                                        permutation=permutation)
                
                

                start_time = time.time()
                # print('Inside main_MBAE the target is', np.argmax(target_label))

                result = attack.attack_batch(input_img, target_label)
                end_time = time.time()

                adv_img, query_count, summary, attack_completed, values_inter = result

                perturbed_img = adv_img


                if not attack_completed:
                    steps_done = values_inter['steps']
                    iteration_scale = values_inter['iteration_scale']
                    ord_domain = values_inter['ord_domain']
                    permutation = values_inter['permutation']
                extremal_pixels=len(adv_img[np.abs(adv_img-img0)>0.02].reshape(-1,))
                len_variables=len(adv_img.reshape(-1,))
                print('Fraction of extremal pixels', extremal_pixels,len_variables,extremal_pixels/len_variables)
                assert np.amax(np.abs(adv_img-img0)) <= FLAGS.eps+1e-3
                assert np.amax(adv_img) <= +0.5+1e-3
                assert np.amin(adv_img) >= -0.5-1e-3

                time_attack += (end_time-start_time)   
            
                if attack_completed:
                    full_adversarial = adv_img
                    print(adv_img.shape)
                    final_pred = np.argmax(model.predict(2*np.array([full_adversarial]))[0])
                    print('Predicted',final_pred)
                    print('Target', np.argmax(target_label))
                    if final_pred == np.argmax(target_label):
                        success_count += 1
                        print('--- SUCCEEEED ----')
                
                

        if not right_classification:
            continue

        frob_norm = np.linalg.norm(adv_img-img0)

        attacks.append([query_count, real_label, np.argmax(target_label), values_inter['loss'],frob_norm])
        with open(saving_dir, "wb") as fp:
                    pickle.dump(attacks, fp)
            # logger.close(num_attempts=total_count)
        print('Number of success = {} / {}.'.format(success_count, total_count))
