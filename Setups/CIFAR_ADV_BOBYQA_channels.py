"""
export PATH=/home/ughi/Documents/Cleaned\ Adversarial\ Attacks/Adversarial_Attacks/bin/:$PATH
export PYTHONPATH=/home/ughi/Documents/Cleaned\ Adversarial\ Attacks/Adversarial_Attacks/lib/python3.6/site-packages/:$PYTHONPATH
"""
# coding: utf-8

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
sys.path.append("./")
import tensorflow as tf
import numpy as np
import random
import time

import pickle

from Setups.Data_and_Model.setup_cifar import CIFAR, CIFARModel
from Setups.Data_and_Model.setup_mnist import MNIST, MNISTModel
# from Setups.Data_and_Model.setup_inception import ImageNet, InceptionModel

from Attack_Code.MNIST_CIFAR_boby_gen import BlackBox
from Attack_Code.BOBYQA.BOBYQA_Attack_Adversary_channels_2 import BlackBox_BOBYQA

dir_path = os.path.dirname(os.path.realpath(__file__))
main_dir = os.path.abspath(os.path.join(dir_path, os.pardir))

from PIL import Image
import torch as ch
from robustness.datasets import CIFAR as CIFAR_robustness
from robustness.model_utils import make_and_restore_model

flags = tf.app.flags
flags.DEFINE_string('dataset', 'cifar10', 'model name')
flags.DEFINE_string('attack', 'boby', 'Attack considered')
flags.DEFINE_integer('test_size', 1, 'Number of test images.')
flags.DEFINE_integer('max_evals', None, 'Maximum number of function evaluations.')
flags.DEFINE_integer('print_every', 10, 'Every iterations the attack function has to print out.')
flags.DEFINE_integer('seed', 1216, 'random seed')
flags.DEFINE_bool('Adversary_trained', True, ' Use the adversarially trained nets')
flags.DEFINE_string('description', '', 'Further description describing the results')

# BOPBYQA parameters
flags.DEFINE_string('interpolation', 'block', 'Interpolation inbetween grod elements in the BOBYQA attack')
flags.DEFINE_integer('batch_size', 20, 'Dimension of the optimisation domain.')
flags.DEFINE_float('eps', 0.1, 'perturbation energy')
flags.DEFINE_bool('use_resize', True, 'if using hierarchical approach')
flags.DEFINE_integer('n_channels', 3, 'n channels in the perturbation grid')
flags.DEFINE_bool('save', True, 'If saving the results')
flags.DEFINE_float('max_f', 1.3 , 'Maximum number of function evaluations in the BOBYQA attack')
flags.DEFINE_string('over', 'over', 'Kind of interpolation within block in the BOBYQA attack')
flags.DEFINE_bool('rounding', True, 'If to include the rounding possibility in the attacks')
FLAGS = flags.FLAGS

def generate_data(data, samples, targeted=True, start=0):
    """
    Generate the input data to the attack algorithm.

    data: the images to attack
    samples: number of samples to use
    targeted: if true, construct targeted attacks, otherwise untargeted attacks
    start: offset into data to use
    inception: if targeted and inception, randomly sample 100 targets intead of 1000
    """
    inputs = []
    targets = []
    labels = []
    true_ids = []
    for i in range(samples):
        if targeted:
            seq = range(data.test_labels.shape[1])
            for j in seq:
                # skip the original image label
                if (j == np.argmax(data.test_labels[start+i])):
                    continue
                inputs.append(data.test_data[start+i])
                targets.append(np.eye(data.test_labels.shape[1])[j])
                labels.append(data.test_labels[start+i])
                true_ids.append(start+i)
        else:
            inputs.append(data.test_data[start+i])
            targets.append(data.test_labels[start+i])
            labels.append(data.test_labels[start+i])
            true_ids.append(start+i)

    inputs = np.array(inputs)
    targets = np.array(targets)
    labels = np.array(labels)
    true_ids = np.array(true_ids)

    return inputs, targets, labels, true_ids


class Model_Class():
    def __init__(self, model):
        self.model = model

    def predict(self, img):
        img = np.moveaxis(img,3,1)
        img = ch.tensor(img).float()
        logit,_ =self.model(img+0.5) 
        return logit.detach().numpy()

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


if __name__ == '__main__':
    # Set Parameters of the attacks
    if FLAGS.max_evals is None:
        if (FLAGS.dataset == 'mnist' or 
                FLAGS.dataset == 'cifar10'):
            FLAGS.max_evals = 3000
        elif FLAGS.dataset == 'inception':
            FLAGS.max_evals = 15000
    
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # Initialise list to save the results
    list_attack = []
    
    # load network
    if FLAGS.dataset == "mnist":
        if FLAGS.Adversary_trained:
            data, model = MNIST(), MNISTModel(main_dir + "/Models/mnist-distilled-100", sess, use_log)
        else:
            data, model = MNIST(), MNISTModel(main_dir + "/Models/mnist", sess, use_log)
    elif FLAGS.dataset == "cifar10":
        ds = CIFAR_robustness('./Data/CIFAR-10')
        data = CIFAR()
        if FLAGS.Adversary_trained:
            model,_ = make_and_restore_model(arch='resnet50', dataset=ds, 
                                    resume_path='./Models/cifar_robust.pt',
                                    parallel=False)
        else:
            model = CIFARModel(main_dir + "/Models/cifar-distilled-100", sess, use_log)
            
    model.eval()
    model = Model_Class(model.float())

    # loading the data
    all_inputs, all_targets, all_labels, all_true_ids = generate_data(data, 
                                samples=FLAGS.test_size, targeted=True,
                                start=0)

    # Set Loading and Saving directories
    if FLAGS.dataset == 'mnist':
        saving_dir = './Results/MNIST/'
    elif FLAGS.dataset == 'cifar10':
        saving_dir = './Results/CIFAR/'
    elif FLAGS.dataset == 'inception':
        saving_dir = './Results/Imagenet/'

    saving_name = (saving_dir+FLAGS.attack +'_adversary_' + str(FLAGS.Adversary_trained) +
                        '_interpolation_' + FLAGS.interpolation +
                        '_eps_'+str(FLAGS.eps) +
                        '_max_eval_'+ str(FLAGS.max_evals) + 
                        '_n_channels_' + str(FLAGS.n_channels) + 
                        '_over_' + str(FLAGS.over) + 
                        '_max_f_' + str(FLAGS.max_f) + 
                        '_rounding_' + str(FLAGS.rounding) + 
                        FLAGS.description + '.txt')  

    
    # Loading the previous results obtained with the same saving directory
    already_done = 0
    # if we are not saving, i.e. doing a test, we want to start from the same image
    if FLAGS.save: 
        if os.path.exists(saving_name):
            if os.path.getsize(saving_name)>0:
                with open(saving_name, "rb") as fp:
                    list_attack = pickle.load(fp)
                already_done = len(list_attack)
    
        if already_done>0:
            found = False
            # set the label and target of the image that has been last attacked
            lab = list_attack[-1][2]
            tar = list_attack[-1][3] 
            # iterate throught the data to find at what point we are
            while not found:
                
                lab_ = np.argmax(all_labels[already_done-1:already_done])
                tar_ = np.argmax(all_targets[already_done-1:already_done])

                already_done +=1
                if lab == lab_ and tar==tar_:
                    found = True
                    already_done -= 1
    
    print('[EXPERIMENTAL SETUP] Attacking', FLAGS.test_size, 'test images with ', FLAGS.eps,' energy')
    print('                    ', already_done, ' attacks have already been conducted.')
    
    for i in range(already_done,FLAGS.test_size):
        inputs = all_inputs[i:i+1]
        targets = all_targets[i:i+1]
        labels = all_labels[i:i+1]
        print('[L1] Image of class ',np.argmax(labels),' targeted to ', np.argmax(targets),'.')

        original_predict = model.predict(inputs)
        original_predict = np.squeeze(original_predict)
        original_prob = np.sort(original_predict)
        original_class = np.argsort(original_predict)
        if original_class[-1] != np.argmax(labels):
            print("skip wrongly classified image no. {}, original class {}, classified as {}".format(
                                i, np.argmax(labels), original_class[-1]))
            continue

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
                indices = np.argmax(targets)
                lll.append(-np.log(probs_[indices] + 1e-10) + np.log(np.sum(probs_)
                                                                    - probs_[indices] + 1e-10))
                preds_l.append(np.argmax(logits_[0]))
                distances.append(np.max(probs_)-probs_[indices])
            
            if only_loss:
                return lll
            else:
                return lll, logits_, distances

        attack_boby = BlackBox_BOBYQA(loss_func, batch_size=FLAGS.batch_size ,
                                  interpolation = FLAGS.interpolation,
                                  n_channels_input=FLAGS.n_channels,
                                  print_every=FLAGS.print_every, use_resize=FLAGS.use_resize, 
                                  eps=FLAGS.eps, max_eval=FLAGS.max_evals,
                                  over=FLAGS.over, rounding=FLAGS.rounding,
                                  max_f=FLAGS.max_f)

        result = attack_boby.attack_batch(inputs, targets)
        adv, eval_costs, summary, Success = result

        if len(adv.shape) == 3:
            adv = adv.reshape((1,) + adv.shape)
        
        adversarial_predict = model.predict(adv)
        adversarial_predict = np.squeeze(adversarial_predict)
        adversarial_prob = np.sort(adversarial_predict)
        adversarial_class = np.argsort(adversarial_predict)

        list_attack.append([eval_costs, adversarial_predict,
                        np.argmax(labels), np.argmax(targets)])
        
        print("[STATS][L1] no={}, success: {}, prev_class = {}, new_class = {}".format(i, Success, original_class[-1], adversarial_class[-1]))
        sys.stdout.flush()

        if FLAGS.save:
            with open(saving_name , "wb") as fp:
                    pickle.dump(list_attack, fp)