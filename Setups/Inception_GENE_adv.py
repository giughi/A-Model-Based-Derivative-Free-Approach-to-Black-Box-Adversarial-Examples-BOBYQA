"""
Author: Moustafa Alzantot (malzantot@ucla.edu)
python Setups/main.py --input_dir=/Data/ImageNet/images --test_size=1 \
    --eps=0.05 --alpha=0.15 --mutation_rate=0.10  \
    --max_steps=100000 --output_dir=attack_outputs \
    --pop_size=6 --target=704 --adaptive=True --resize_dim=96
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import sys
sys.path.append("./")
# main_dir = os.path.dirname(os.path.dirname(__file__))

import random
import numpy as np

import pickle
import tensorflow as tf   

from Setups.Data_and_Model.setup_inception import ImageNet, InceptionModel
import Attack_Code.GenAttack.utils as utils
from Attack_Code.GenAttack.genattack_tf2_adv_inception import GenAttack2
import os
from keras import backend as K, regularizers

from tensorflow.contrib.slim.nets import inception
slim = tf.contrib.slim
from tensorflow.python.framework import ops
from tensorflow.python.training import monitored_session
from tensorflow.python.training import saver as tf_saver


dir_path = os.path.dirname(os.path.realpath(__file__))
main_dir = os.path.abspath(os.path.join(dir_path, os.pardir))
checkpoint_path_ = main_dir + '/Models/adv_inception_v3.ckpt'


flags = tf.app.flags
flags.DEFINE_string('input_dir', '', 'Path for input images.')
flags.DEFINE_string('output_dir', 'output', 'Path to save results.')
flags.DEFINE_integer('test_size', 1000, 'Number of test images.')
flags.DEFINE_bool('verbose', True, 'Print logs.')
flags.DEFINE_integer('test_example', None, 'Test only one image')

flags.DEFINE_float('mutation_rate', 0.005, 'Mutation rate')
flags.DEFINE_float('eps', 0.10, 'maximum L_inf distance threshold')
flags.DEFINE_float('alpha', 0.20, 'Step size')
flags.DEFINE_integer('pop_size', 6, 'Population size')
flags.DEFINE_integer('max_steps', 3000, 'Maximum number of iterations')
flags.DEFINE_integer('resize_dim', None, 'Reduced dimension for dimensionality reduction')
flags.DEFINE_bool('adaptive', True, 'Turns on the dynamic scaling of mutation prameters')
flags.DEFINE_string('model', 'inception', 'model name')
flags.DEFINE_integer('target', None, 'target class. if not provided will be random')
flags.DEFINE_integer('seed', 1216, 'random seed')
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

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

if __name__ == '__main__':

    random.seed(FLAGS.seed)
    tf.set_random_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    
    dataset = ImageNet(FLAGS.input_dir)
    inputs, targets, reals, paths = utils.generate_data(dataset, FLAGS.test_size)
    
    total_count = 0 # Total number of images attempted
    success_count = 0

    attacks = []

    num_valid_images = 1000#FLAGS.test_size#len(inputs)
    if FLAGS.worst_attack:
        saving_dir = main_dir+'/Results/Imagenet/ADV_GENE_'+str(FLAGS.eps)+'_2nd_500_WORST_ATTACK_2.txt'
    else:
        saving_dir = main_dir+'/Results/Imagenet/ADV_GENE_'+str(FLAGS.eps)+'_2nd_500_2.txt'
    already_done = 0
    if os.path.exists(saving_dir):
        if os.path.getsize(saving_dir)>0:
            with open(saving_dir, "rb") as fp:
                attacks = pickle.load(fp)
                already_done = len(attacks)

    print('Already done ==============',already_done)

    for ii in range(already_done, num_valid_images):
        if (FLAGS.test_example and FLAGS.test_example != ii):
            continue
        input_img = inputs[ii]
        input_img_path =  paths[ii]
        if FLAGS.target:
            target_label = FLAGS.target + 1
        else:
            target_label = np.argmax(targets[ii])
        real_label = reals[ii]
        
        tf.reset_default_graph()
        # tf.set
        K.clear_session()
        tf.GraphDef().Clear()

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

        # session_conf = tf.ConfigProto(
        #       intra_op_parallelism_threads=1,
        #       inter_op_parallelism_threads=1)
        # sess = tf.Session(config=session_conf)

        with monitored_session.MonitoredSession(
            session_creator=session_creator, hooks=None) as sess:

            model = Adv_Inception(sess)
                
            logits_in1 = model.predict(np.array([inputs[0]]))
            
            logits_in = sess.run(logits, feed_dict={tf_image:np.array([inputs[0]]) * 2.0})
                        
            tf.set_random_seed(FLAGS.seed)

            orig_pred = np.argmax(sess.run(logits, feed_dict={tf_image: 2*[input_img]})[0])
            if FLAGS.verbose:
                print('Real = {}, Predicted = {}, Target = {}'.format(
                    real_label, orig_pred, target_label))
            if FLAGS.worst_attack:
                worst_class = np.argmin(logits_in)
                if worst_class != np.argmin(logits_in1):
                    Warning('THE MODEL IS BEHAVING WIERDLY')
                target_label =worst_class
                print('Because of Worst Case target is', worst_class)
            if orig_pred != real_label:
                if FLAGS.verbose:
                    print('\t Skipping incorrectly classified image.')
                continue

            def loss_func(pert, only_loss=False):
                    nn= len(pert)
                    if nn==299:
                        pert = [pert]
                        nn = 1
                    lll = []
                    preds_l = []
                    distances = []
                    
                    for i in range(nn):
                        logits_ = model.predict([pert[i]])
                        probs_ = softmax(logits_[0])
                        indices = np.argmax(target_label)
                        lll.append(np.log(probs_[indices] + 1e-10) - np.log(np.sum(probs_)- probs_[indices] + 1e-10))
                        preds_l.append(np.argmax(logits_[0]))
                        distances.append(-np.max(probs_)+probs_[indices])
                    
                    if only_loss:
                        return lll
                    else:
                        return softmax(logits_), lll, np.array(distances)

            attack = GenAttack2(loss_f = loss_func,
                    pop_size=FLAGS.pop_size,
                    mutation_rate = FLAGS.mutation_rate,
                    eps=FLAGS.eps,
                    max_steps=FLAGS.max_steps,
                    alpha=FLAGS.alpha,
                    resize_dim=FLAGS.resize_dim,
                    adaptive=FLAGS.adaptive)
        
            
            total_count += 1
            start_time = time.time()
            result = attack.attack(input_img, target_label)
            end_time = time.time()
            attack_time = (end_time-start_time)
            if result is not None:
                success, adv_img, query_count = result
                # print(query_count,adv_img.shape)
                if success:
                    final_pred = np.argmax(model.predict(np.array([adv_img]))[0])
                    if (final_pred == target_label):
                        success_count += 1
                        print('--- SUCCEEEED ----')

                        # logger.add_result(ii, input_img, adv_img, real_label,
                        #     target_label, query_count, attack_time, margin_log)

        frob_norm = np.linalg.norm(adv_img-input_img)
        attacks.append([query_count, real_label, target_label, frob_norm])
        with open(saving_dir, "wb") as fp:
                    pickle.dump(attacks, fp)
        # logger.close(num_attempts=total_count)
        print('Number of success = {} / {}.'.format(success_count, total_count))
    # logger.close(num_attempts=total_count)
    # print('Number of success = {} / {}.'.format(success_count, total_count))
