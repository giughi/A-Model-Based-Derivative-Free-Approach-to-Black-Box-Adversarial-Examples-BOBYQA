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

import tensorflow as tf   
from Attack_Code.BOBYQA.BOBYQA_Attack import BlackBox_BOBYQA
from Setups.Data_and_Model.setup_inception_tensorflow import ImageNet, InceptionModel
import pickle
from keras import backend as K, regularizers
import Attack_Code.GenAttack.utils as utils

flags = tf.app.flags
flags.DEFINE_string('input_dir', '', 'Path for input images.')
flags.DEFINE_string('output_dir', 'output', 'Path to save results.')
flags.DEFINE_integer('test_size', 1000, 'Number of test images.')
flags.DEFINE_bool('verbose', True, 'Print logs.')

flags.DEFINE_integer('test_example', None, 'Test only one image')
flags.DEFINE_integer('max_queries', 15000, 'Maximum number of iterations')
flags.DEFINE_float('eps', 0.10, 'maximum L_inf distance threshold')

# BOBYQA variables 
flags.DEFINE_integer('batch', 50, 'Dimension of the sampling domain.')
flags.DEFINE_string('interp', 'over', 'kind of interpolation done on the data, either  <<over>> or <<linear>>')
flags.DEFINE_integer('seed', 1216, 'random seed')
flags.DEFINE_integer('target', None, 'target class. if not provided will be random')
FLAGS = flags.FLAGS


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

    # Check how many attacks have already been completed (We usualy consider the same seed thus we don't
    # want to repeat any attack)
    saving_dir = main_dir+'/Results/Imagenet/BOBY_'+str(FLAGS.eps)+'.txt'
    already_done = 0
    if os.path.exists(saving_dir):
        if os.path.getsize(saving_dir)>0:
            with open(saving_dir, "rb") as fp:
                attacks = pickle.load(fp)
                already_done = len(attacks) + 1
    print('=====> Already done ', already_done)
    # print('IS GPU AVAILABLE', tf.test.is_gpu_available())
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    for ii in range(already_done, num_valid_images):
        # because of how Tensorflow1 works, we have to reset the session every time we do more than 
        # say 1500 queries. This is due to the architecure saving each for loop and thus slowing down too
        # much the algorithm
        attack_completed = False
        
        input_img = inputs[ii]
        img0 = input_img.copy()
        input_img_path = paths[ii]
        print(type(input_img))

        if FLAGS.target:
            target_label = np.eye(len(targets[ii]))[FLAGS.target + 1]
        else:
            target_label = targets[ii]
        real_label = reals[ii]

        # This is the initialisiation of the variables that will be saved each time that the session 
        # has to be restarted
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
            K.clear_session()
            tf.GraphDef().Clear()

            # session_conf = tf.ConfigProto(intra_op_parallelism_threads=5,
            #                               inter_op_parallelism_threads=5)

            # with tf.Session(config=session_conf) as sess:
            with tf.Session() as sess:
                tf.set_random_seed(FLAGS.seed)
                model = InceptionModel(sess, use_log=True)
                test_in = tf.placeholder(tf.float32, (1, 299, 299, 3), 'x')
                logit_pred = model.predict(test_in)
                test_pred = tf.argmax(logit_pred, axis=1)
                
                attack = BlackBox_BOBYQA(sess, model, batch_size=FLAGS.batch , max_iterations=10000,  
                                        confidence=0, targeted=True, use_resize=True, 
                                        L_inf=FLAGS.eps, max_eval=FLAGS.max_queries, print_every=1,
                                        done_eval_costs=query_count, max_eval_internal=15000, 
                                        perturbed_img=perturbed_img, ord_domain=ord_domain, steps_done=steps_done,
                                        iteration_scale=iteration_scale,image0=img0, over=FLAGS.interp, 
                                        permutation=permutation)
                
                # We first check that the image is correctly classified by the net. Otherwise we skip it
                if first_iter:
                    orig_pred = sess.run(test_pred,  feed_dict={test_in: [input_img]})[0]
                    logit_pr = sess.run(logit_pred,  feed_dict={test_in: [input_img]})[0]
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

                start_time = time.time()

                result = attack.attack_batch(input_img, target_label)
                end_time = time.time()

                adv_img, query_count, summary, attack_completed, values_inter = result

                perturbed_img = adv_img


                if not attack_completed:
                    # save the variable for the next session
                    steps_done = values_inter['steps']
                    iteration_scale = values_inter['iteration_scale']
                    ord_domain = values_inter['ord_domain']
                    permutation = values_inter['permutation']
                # Check that the constraints are satified
                assert np.amax(np.abs(adv_img-img0)) <= FLAGS.eps+1e-3
                assert np.amax(adv_img) <= +0.5+1e-3
                assert np.amin(adv_img) >= -0.5-1e-3

                time_attack += (end_time-start_time)   
            
                if attack_completed:
                    full_adversarial = adv_img
                    print(adv_img.shape)
                    final_pred = sess.run(test_pred, feed_dict={test_in: [full_adversarial]})[0]
                    print('Predicted',final_pred)
                    print('Target', np.argmax(target_label))
                    if final_pred == np.argmax(target_label):
                        success_count += 1
                        print('--- SUCCEEEED ----')
                
                sess.close()
                del model
                del attack

        if not right_classification:
            # This allows to skip the images when wrongly classified
            continue

        frob_norm = np.linalg.norm(adv_img-img0)


        attacks.append([query_count, real_label, np.argmax(target_label), values_inter['loss'],frob_norm])
        with open(saving_dir, "wb") as fp:
                    pickle.dump(attacks, fp)
        print('Number of success = {} / {}.'.format(success_count, total_count))
