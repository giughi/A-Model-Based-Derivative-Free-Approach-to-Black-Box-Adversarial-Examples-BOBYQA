"""
Author: Giuseppe Ughi

This file is against the inheritance of weight 

python Setups/main_MBA3_2.py --input_dir=/images/ --test_size=1 \
    --eps=0.05 \
    --max_steps=500000 --output_dir=attack_outputs  \
    --target=704
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

import tensorflow as tf   
from Attack_Code.BOBYQA.BOBYQA_Attack_rand_multi_with_restarts import BlackBox_BOBYQA
from Setups.Data_and_Model.setup_inception_2 import ImageNet, InceptionModel
import pickle
from keras import backend as K, regularizers
from Attack_Code.GenAttack import utils

flags = tf.app.flags
flags.DEFINE_string('input_dir', '', 'Path for input images.')
flags.DEFINE_string('output_dir', 'output', 'Path to save results.')
flags.DEFINE_integer('test_size', 1, 'Number of test images.')
flags.DEFINE_bool('verbose', True, 'Print logs.')
flags.DEFINE_integer('batch', 50, 'Dimension of the sampling domain.')
flags.DEFINE_integer('test_example', None, 'Test only one image')

flags.DEFINE_float('eps', 0.10, 'maximum L_inf distance threshold')
flags.DEFINE_integer('max_steps', 10000, 'Maximum number of iterations')
flags.DEFINE_integer('resize_dim', 96, 'Reduced dimension for dimensionality reduction')
flags.DEFINE_string('model', 'inception', 'model name')
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
#         logger = utils.ResultLogger(FLAGS.output_dir, FLAGS.flag_values_dict())
    for ii in range(num_valid_images):
        
        attack_completed = False
        
        input_img = inputs[ii]
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

        right_classification = True
        first_iter = True


        while not attack_completed:
            
            time_attack = 0
            tf.reset_default_graph()
            # tf.set
            K.clear_session()
            tf.GraphDef().Clear()

            with tf.Session() as sess:
                tf.set_random_seed(FLAGS.seed)
                print('Sess',sess)
                model = InceptionModel(sess, use_log=True)
                test_in = tf.placeholder(tf.float32, (1, 299, 299, 3), 'x')
                test_pred = tf.argmax(model.predict(test_in), axis=1)
                
                attack = BlackBox_BOBYQA(sess, model, batch_size=50, max_iterations=10000, print_every=10, 
                                        early_stop_iters=100, confidence=0, targeted=True, use_log=True,
                                        use_tanh=False, use_resize=True, L_inf=FLAGS.eps, rank=3, ordered_domain=True,
                                        image_distribution=True, mixed_distributions=False, max_eval=15000,
                                        done_eval_costs=query_count, max_eval_internal=1000, 
                                        perturbed_img=perturbed_img, ord_domain=ord_domain, steps_done=steps_done,
                                        iteration_scale=iteration_scale)
            
                if FLAGS.test_example and FLAGS.test_example != ii:
                    continue
                
                if first_iter:
                    orig_pred = sess.run(test_pred, feed_dict={test_in: [input_img]})[0]
                    if FLAGS.verbose:
                        print('Real = {}, Predicted = {}, Target = {}'.format(
                            real_label, orig_pred, target_label))
                    if orig_pred != real_label:
                        if FLAGS.verbose:
                            print('\t Skipping incorrectly classified image.')
                        right_classification = False
                        break
                    total_count += 1
                    first_iter=False

                start_time = time.time()
                print('Inside main_MBAE the target is', np.argmax(target_label))
                result = attack.attack_batch(input_img, target_label)
                end_time = time.time()

                adv_img, query_count, summary, attack_completed, values_inter = result
                
                perturbed_img = adv_img


                if not attack_completed:
                    steps_done = values_inter['steps']
                    iteration_scale = values_inter['iteration_scale']
                    ord_domain = values_inter['ord_domain']

                time_attack += (end_time-start_time)   
            
                if attack_completed:
                    # adv_img, query_count, margin_log = result
                    # summary.to_pickle('./Results/Imagenet/Summary_in_np1_max_nxn')
                    full_adversarial = adv_img
                    print(adv_img.shape)
                    # np.save('adv_image', adv_img)
                    # print(input_img.shape)
                    final_pred = sess.run(test_pred, feed_dict={test_in: [full_adversarial]})[0]
                    print('Predicted',final_pred)
                    print('Target', np.argmax(target_label))
                    if final_pred == np.argmax(target_label):
                        success_count += 1
                        print('--- SUCCEEEED ----')
                        # logger.add_result(ii, input_img, adv_img, real_label,
                        #     target_label, query_count, attack_time, margin_log)
                
                sess.close()
                del model
                del attack

        if not right_classification:
            continue

        attacks.append([query_count, real_label, np.argmax(target_label)])
        with open(main_dir+'/Results/Imagenet/BOBY_'+str(FLAGS.eps)+'.txt', "wb") as fp:
                    pickle.dump(attacks, fp)
        # logger.close(num_attempts=total_count)
        print('Number of success = {} / {}.'.format(success_count, total_count))
