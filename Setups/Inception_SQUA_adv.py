"""
"""
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
from tensorflow.contrib.slim.nets import inception
slim = tf.contrib.slim
from tensorflow.python.framework import ops
from tensorflow.python.training import monitored_session
from tensorflow.python.training import saver as tf_saver


from Setups.Data_and_Model.setup_inception_tensorflow import ImageNet, InceptionModel
from Attack_Code.Square_Attack import utils
from Attack_Code.GenAttack import utils as utils_2
import pickle
from keras import backend as K, regularizers
from datetime import datetime

flags = tf.app.flags
flags.DEFINE_string('input_dir', '', 'Path for input images.')
flags.DEFINE_string('output_dir', 'output', 'Path to save results.')
flags.DEFINE_integer('test_size', 600, 'Number of test images.')
flags.DEFINE_bool('verbose', True, 'Print logs.')
flags.DEFINE_integer('batch', 50, 'Dimension of the sampling domain.')
flags.DEFINE_integer('test_example', None, 'Test only one image')

flags.DEFINE_float('eps', 0.10, 'maximum L_inf distance threshold')
flags.DEFINE_integer('max_steps', 15000, 'Maximum number of iterations')
flags.DEFINE_integer('resize_dim', 96, 'Reduced dimension for dimensionality reduction')
flags.DEFINE_string('model', 'inception', 'model name')
flags.DEFINE_string('interp', 'over', 'kind of interpolation done on the data, either  <<over>> or <<linear>>')
flags.DEFINE_integer('seed', 1216, 'random seed')
flags.DEFINE_integer('target', None, 'target class. if not provided will be random')
flags.DEFINE_bool('targeted', True, 'Bool on targeted attack.')
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


def p_selection(p_init, it, n_iters):
    """ Piece-wise constant schedule for p (the fraction of pixels changed on every iteration). """
    it = int(it / n_iters * 10000)

    if 10 < it <= 50:
        p = p_init / 2
    elif 50 < it <= 200:
        p = p_init / 4
    elif 200 < it <= 500:
        p = p_init / 8
    elif 500 < it <= 1000:
        p = p_init / 16
    elif 1000 < it <= 2000:
        p = p_init / 32
    elif 2000 < it <= 4000:
        p = p_init / 64
    elif 4000 < it <= 6000:
        p = p_init / 128
    elif 6000 < it <= 8000:
        p = p_init / 256
    elif 8000 < it <= 10000:
        p = p_init / 512
    else:
        p = p_init

    return p

def softmax_square(x):
    e_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return e_x / e_x.sum(axis=0, keepdims=True)

def loss(y, logits, targeted=True, cross_entropy=False):
        """ Implements the margin loss (difference between the correct and 2nd best class). """
        # print(len(y), len(logits))
        if not cross_entropy:
            if targeted:
                targ_loss = logits[y]
                preds_correct_class = np.max(logits)
                diff = preds_correct_class  - targ_loss  # difference between the correct class and all other classes
                return diff
            else:
                preds_correct_class = np.sort(logits)
                diff = logits[y][0] - preds_correct_class[-2]  # difference between the correct class and all other classes
                return np.array([diff])
        else:            
            probs = logits#softmax_square(logits)
            loss_ = -np.log((probs*y).sum()+1e-10) + np.log(np.sum(probs)- (probs*y).sum() + 1e-10)
            loss_ = loss_ * -1 if not targeted else loss_
            return loss_.flatten()

def square_attack_linf(sess, model, x, y, eps, n_iters, p_init,targeted):
    """ The Linf square attack """
    np.random.seed(0)  # important to leave it here as well
    min_val, max_val = -0.5, 0.5 
    h, w, c = x.shape[1:]
    n_features = c*h*w
    n_ex_total = x.shape[0]

    test_in = tf.placeholder(tf.float32, (1, 299, 299, 3), 'x')
    test_pred = model.predict(test_in)
                    
        
    # Vertical stripes initialization
    x_best = np.clip(x + np.random.choice([-eps, eps], size=[x.shape[0], 1, w, c]), min_val, max_val)
    logits = sess.run(test_pred, feed_dict={test_in: x_best})[0]
    margin_min = loss(y[0], logits,targeted)
    loss_min = loss(y[0], logits,targeted, cross_entropy=True)
    n_queries = np.ones(x.shape[0])  # ones because we have already used 1 query

    time_start = time.time()
    # metrics = np.zeros([n_iters, 7])
    for i_iter in range(n_iters - 1):
        idx_to_fool = margin_min > 0
        x_curr, x_best_curr = x, x_best
        y_curr, margin_min_curr, loss_min_curr = y, margin_min, loss_min
        deltas = x_best_curr - x_curr

        p = p_selection(p_init, i_iter, n_iters)
        for i_img in range(x_best_curr.shape[0]):
            s = int(round(np.sqrt(p * n_features / c)))
            s = min(max(s, 1), h-1)  # at least c x 1 x 1 window is taken and at most c x h-1 x h-1
            center_h = np.random.randint(0, h - s)
            center_w = np.random.randint(0, w - s)

            x_curr_window = x_curr[i_img, center_h:center_h+s, center_w:center_w+s, :]
            x_best_curr_window = x_best_curr[i_img, center_h:center_h+s, center_w:center_w+s, :]
            # prevent trying out a delta if it doesn't change x_curr (e.g. an overlapping patch)
            while np.sum(np.abs(np.clip(x_curr_window + deltas[i_img, center_h:center_h+s, center_w:center_w+s, :], 
                                        min_val, max_val)
                                - x_best_curr_window) 
                         < 10**-7) == c*s*s:
                # the updates are the same across all elements in the square
                deltas[i_img, center_h:center_h+s, center_w:center_w+s, :] = np.random.choice([-eps, eps], size=[1, 1, c])

        x_new = np.clip(x_curr + deltas, min_val, max_val)

        logits = sess.run(test_pred, feed_dict={test_in: x_new})[0]
        loss_val = loss(y[0], logits,targeted, cross_entropy=True)
        margin = loss(y_curr[0], logits,targeted)
        # print(margin)
        idx_improved = loss_val < loss_min_curr #margin < margin_min_curr

        loss_min = (idx_improved) * loss_val + (~idx_improved) * loss_min_curr
        margin_min = idx_improved * margin + ~idx_improved * margin_min_curr
        # margin_min[idx_to_fool] = idx_improved * margin + ~idx_improved * margin_min_curr
        idx_improved = np.reshape(idx_improved, [-1, *[1]*len(x.shape[:-1])])
        x_best = idx_improved * x_new + ~idx_improved * x_best_curr
        n_queries += 1
        # x_best[idx_to_fool] = idx_improved * x_new + ~idx_improved * x_best_curr
        # n_queries[idx_to_fool] += 1

        # different metrics to keep track of
        acc = margin_min[0]
        # acc_corr = (margin_min > 0.0).mean()
        # mean_nq, mean_nq_ae, median_nq_ae = np.mean(n_queries), np.mean(n_queries[margin_min <= 0]), np.median(n_queries[margin_min <= 0])
        time_total = time.time() - time_start
        print('[L1] {}: loss ={:.3f}, margin={}  (n_ex={}, eps={:.3f}, {:.2f}s)'.
            format(i_iter+1, loss_min[0] ,acc, x.shape[0], eps, time_total))

        if acc<=0:
            break
    return n_queries, x_best


if __name__ == '__main__':

    attacks_done = []
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    dataset = ImageNet(FLAGS.input_dir)
    inputs, targets, reals, paths = utils_2.generate_data(dataset, FLAGS.test_size)
    
    num_valid_images = FLAGS.test_size  # len(inputs)
    total_count = 0  # Total number of images attempted
    success_count = 0
    attacks = []
    log_querries = []
#         logger = utils.ResultLogger(FLAGS.output_dir, FLAGS.flag_values_dict())
    
    saving_dir = main_dir+'/Results/Imagenet/ADV_SQUA_'+str(FLAGS.eps)+'_CE.txt'
    print(saving_dir)
    saving_dir_full = main_dir+'/Results/Imagenet/ADV_SQUA_'+str(FLAGS.eps)+'_CE.txt'
    already_done = 0
    if os.path.exists(saving_dir_full):
        if os.path.getsize(saving_dir_full)>0:
            with open(saving_dir_full, "rb") as fp:
                attacks = pickle.load(fp)
                already_done = len(attacks)
    print('=====> Already done ', already_done)

    for ii in range(already_done, num_valid_images):
        
        attack_completed = False
        
        input_img = np.array([inputs[ii]])
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

            session_conf = tf.ConfigProto(intra_op_parallelism_threads=8,
                                          inter_op_parallelism_threads=8)

            session_conf = tf.ConfigProto(gpu_options=tf.GPUOptions(visible_device_list= '0', 
                                            per_process_gpu_memory_fraction=0.20))


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
            # with tf.Session(config=session_conf) as sess:
            #     tf.set_random_seed(FLAGS.seed)
                # print('Sess',sess)
                
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

                
                # # test_in = tf.placeholder(tf.float32, (1, , 299, 3), 'x')
                # test_pred = tf.argmax(logits, axis=1)
                # if first_iter:
                #     orig_pred = sess.run(test_pred, feed_dict={tf_image: input_img*2})[0]
                #     if FLAGS.verbose:
                #         print('Real = {}, Predicted = {}, Target = {}'.format(
                #             real_label, orig_pred, np.argmax(target_label)))
                #     if orig_pred != real_label:
                #         if FLAGS.verbose:
                #             print('\t Skipping incorrectly classified image.')
                #         right_classification = False
                #         break
                #     total_count += 1
                #     first_iter=False

                start_time = time.time()
                # print('Inside main_MBAE the target is', np.argmax(target_label))
                if FLAGS.targeted:
                    y_target_onehot = utils.dense_to_onehot(np.array([np.argmax(target_label)]), n_cls=len(target_label))
                else:
                    y_target_onehot = utils.dense_to_onehot(np.array([real_label]), n_cls=len(target_label))
                result = square_attack_linf(sess, model, img0, y_target_onehot, FLAGS.eps, FLAGS.max_steps, 0.01,FLAGS.targeted)
                
                end_time = time.time()

                query_count, adv_img = result
                
                attack_completed = True

                perturbed_img = adv_img
                
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
                    final_pred = sess.run(test_pred, feed_dict={test_in: full_adversarial})[0]
                    print('Predicted',final_pred)
                    print('Target', np.argmax(target_label))
                    if final_pred == np.argmax(target_label) and FLAGS.targeted:
                        success_count += 1
                        print('--- SUCCEEEED ----')
                    elif final_pred != real_label and not FLAGS.targeted:
                        success_count += 1
                        print('--- SUCCEEEED ----')
                
                sess.close()
                del model
                # del attack

        if not right_classification:
            continue
        

        attacks.append([query_count, real_label, np.argmax(target_label)])
        with open(saving_dir, "wb") as fp:
                    pickle.dump(attacks, fp)
        log_querries = np.append(log_querries, query_count)
        # logger.close(num_attempts=total_count)
        print('Number of success = {} / {} with median {}'.format(success_count, total_count, np.median(log_querries)))
