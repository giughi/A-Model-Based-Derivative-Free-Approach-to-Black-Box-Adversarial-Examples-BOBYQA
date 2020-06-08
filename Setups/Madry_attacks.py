"""
conda activate Adv_Attacks_cpu
cd ./Documents/GITBOBY/A-Model-Based-Derivative-Free-Approach-to-Black-Box-Adversarial-Examples-BOBYQA/
python Setups/Madry_attacks.py --attack=square --max_f=1.3 --rounding=True --test_size=1200 --save=True  --dataset=cifar10 --subspace_attack=True --subspace_dimension=1000 --Adversary_trained=False --eps=0.05

conda activate Adv_Attacks_source_pytorch
cd ./Documents/GITBOBY/A-Model-Based-Derivative-Free-Approach-to-Black-Box-Adversarial-Examples-BOBYQA/
python Setups/Madry_attacks.py --attack=square --max_f=1.3 --rounding=True --test_size=300 --save=False --dataset=ImageNet --subspace_attack=True --subspace_dimension=1000 --Adversary_trained=True --eps=0.05
"""
# coding: utf-8

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
sys.path.append("./")
import tensorflow as tf
import torchvision.models as models_ch
import torch as ch
import numpy as np
import random
import time

import pickle

from Setups.Data_and_Model.setup_cifar import CIFAR, CIFARModel
from Setups.Data_and_Model.setup_mnist import MNIST, MNISTModel
from Setups.Data_and_Model.setup_inception_2 import ImageNet
from Setups.Data_and_Model.wrapper_model_loss_f import wrapper_model, wrapper_loss

from Attack_Code.MNIST_CIFAR_boby_gen import BlackBox
from Attack_Code.BOBYQA.BOBYQA_Attack_Adversary_channels_2 import BlackBox_BOBYQA
from Attack_Code.Combinatorial.attacks.parsimonious_attack_madry import ParsimoniousAttack
from Attack_Code.Square_Attack.attack_madry import square_attack_linf

dir_path = os.path.dirname(os.path.realpath(__file__))
main_dir = os.path.abspath(os.path.join(dir_path, os.pardir))

from PIL import Image
# import torch as ch
from robustness.datasets import CIFAR as CIFAR_robustness
from robustness.datasets import ImageNet as Imagenet_robustness
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
flags.DEFINE_float('eps', 0.1, 'perturbation energy')
flags.DEFINE_integer('batch_size', None, 'Dimension of the optimisation domain.')
flags.DEFINE_bool('subspace_attack', False, ' Attack only a fixed number of pixels with highest variability')
flags.DEFINE_integer('subspace_dimension', None, 'Dimension of the subspace optimisation domain when doing subspace attack.')

# BOPBYQA parameters
flags.DEFINE_string('interpolation', 'block', 'Interpolation inbetween grod elements in the BOBYQA attack')
flags.DEFINE_bool('use_resize', True, 'if using hierarchical approach')
flags.DEFINE_integer('n_channels', 3, 'n channels in the perturbation grid')
flags.DEFINE_bool('save', True, 'If saving the results')
flags.DEFINE_float('max_f', 1.3 , 'Maximum number of function evaluations in the BOBYQA attack')
flags.DEFINE_string('over', 'over', 'Kind of interpolation within block in the BOBYQA attack')
flags.DEFINE_bool('rounding', True, 'If to include the rounding possibility in the attacks')

# COMBI parameters
flags.DEFINE_string('asset_dir', default=main_dir+'/Attack_Code/Combinatorial/assets', help='directory assets')
flags.DEFINE_bool('targeted', default=True, help='bool on targeted')
flags.DEFINE_integer('max_iters', 1, help='maximum iterations') 
flags.DEFINE_integer('block_size', default=128, help='blck size') 
flags.DEFINE_bool('no_hier', default=False, help='bool on hierarchical attack') #########
flags.DEFINE_integer('dim_image', default=32, help='Dimension of the image that we feed as an input')
flags.DEFINE_integer('num_channels', default=3, help='Channels of the image that we feed as an input')

# SQUARE parameters
flags.DEFINE_float('p_init', 0.1 , 'dimension of the blocks')

FLAGS = flags.FLAGS

def generate_data(data, samples, dataset, targeted=True, start=0):
    """
    Generate the input data to the attack algorithm.

    data: the images to attack
    samples: number of samples to use
    targeted: if true, construct targeted attacks, otherwise untargeted attacks
    start: offset into data to use
    """
    inputs = []
    targets = []
    labels = []
    for i in range(samples):
        if dataset == 'cifar10':
            seq = range(data.test_labels.shape[1])
            for j in seq:
                # skip the original image label
                if (j == np.argmax(data.test_labels[start+i])):
                    continue
                inputs.append(data.test_data[start+i])
                targets.append(np.eye(data.test_labels.shape[1])[j])
                labels.append(data.test_labels[start+i])
        elif dataset == 'ImageNet':
            num_labels = data.test_labels.shape[1]
            inputs.append(data.test_data[i])
            labels.append(data.test_labels[i])
            other_labels = [x for x in range(num_labels) if data.test_labels[i][x] == 0]
            random_target = [0 for _ in range(num_labels)]
            random_target[np.random.choice(other_labels)] = 1
            targets.append(random_target)

    inputs = np.array(inputs)
    targets = np.array(targets)
    labels = np.array(labels)

    return inputs, targets, labels


if __name__ == '__main__':
    # Set Parameters of the attacks
    if FLAGS.max_evals is None:
        if (FLAGS.dataset == 'mnist' or 
                FLAGS.dataset == 'cifar10'):
            FLAGS.max_evals = 3000
        elif FLAGS.dataset == 'ImageNet':
            FLAGS.max_evals = 15000
    
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # Initialise list to save the results
    list_attack = []
    single_output = False # Need this to deal wiht the different net for non adversary
                          # trained net with imagenet
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
            model,_ = make_and_restore_model(arch='resnet50', dataset=ds, 
                                    resume_path='./Models/cifar_nat.pt',
                                    parallel=False)
    elif FLAGS.dataset == "ImageNet":
        ds = Imagenet_robustness('./Data/ImageNet/images')
        data = ImageNet('', dimension=299)
        if FLAGS.Adversary_trained:
            model,_ = make_and_restore_model(arch='resnet50', dataset=ds, 
                                    resume_path='./Models/imagenet_linf_8.pt',
                                    parallel=False)
        else:
            model = models_ch.resnet50(pretrained=True)
            single_output=True
    
    if ch.cuda.is_available():
        model.eval()
        model.to('cuda')
        # model.eval()
        model = wrapper_model(model.float(), FLAGS.attack, single_output, cuda=True)
    else:
        model.eval()
        model = wrapper_model(model.float(), FLAGS.attack, single_output)
    # loading the data
    all_inputs, all_targets, all_labels = generate_data(data, dataset=FLAGS.dataset,
                                samples=FLAGS.test_size, targeted=True,
                                start=0)

    # Set Loading and Saving directories
    if FLAGS.dataset == 'mnist':
        saving_dir = './Results/MNIST/'
    elif FLAGS.dataset == 'cifar10':
        saving_dir = './Results/CIFAR/'
    elif FLAGS.dataset == 'ImageNet':
        saving_dir = './Results/Imagenet/'
    
    if FLAGS.attack == 'boby':
        if FLAGS.batch_size is None:
            FLAGS.batch_size = 20
        saving_name = (saving_dir+FLAGS.attack +'_adversary_' + str(FLAGS.Adversary_trained) +
                            '_interpolation_' + FLAGS.interpolation +
                            '_eps_'+str(FLAGS.eps) +
                            '_max_eval_'+ str(FLAGS.max_evals) + 
                            '_n_channels_' + str(FLAGS.n_channels) + 
                            '_over_' + str(FLAGS.over) + 
                            '_max_f_' + str(FLAGS.max_f) + 
                            '_rounding_' + str(FLAGS.rounding) + 
                            '_subspace_attack_' + str(FLAGS.subspace_attack) +
                            '_subspace_dimension_' + str(FLAGS.subspace_dimension) +
                            FLAGS.description + '.txt')  
    elif FLAGS.attack == 'combi':
        if FLAGS.batch_size is None:
            FLAGS.batch_size = 64
        saving_name = (saving_dir+FLAGS.attack +'_adversary_' + str(FLAGS.Adversary_trained) +
                            '_eps_'+str(FLAGS.eps) +
                            '_max_eval_'+ str(FLAGS.max_evals) + 
                            '_max_iters_' + str(FLAGS.max_iters) + 
                            '_block_size_' + str(FLAGS.block_size) +
                            '_batch_size_' + str(FLAGS.batch_size) +
                            '_no_hier_' + str(FLAGS.no_hier) + 
                            '_subspace_attack_' + str(FLAGS.subspace_attack) +
                            '_subspace_dimension_' + str(FLAGS.subspace_dimension) +
                            FLAGS.description + '.txt')  
    elif FLAGS.attack == 'square':
        saving_name = (saving_dir+FLAGS.attack +'_adversary_' + str(FLAGS.Adversary_trained) +
                            '_eps_'+str(FLAGS.eps) +
                            '_max_eval_'+ str(FLAGS.max_evals) + 
                            '_p_init_' + str(FLAGS.p_init) +
                            '_subspace_attack_' + str(FLAGS.subspace_attack) +
                            '_subspace_dimension_' + str(FLAGS.subspace_dimension) +
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
        if FLAGS.attack == 'combi':
            inputs = inputs[0]
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

        loss_func = wrapper_loss(FLAGS.attack, targets, model)
        if FLAGS.attack=='boby':
            attack = BlackBox_BOBYQA(loss_func, batch_size=FLAGS.batch_size ,
                                    interpolation = FLAGS.interpolation,
                                    n_channels_input=FLAGS.n_channels,
                                    print_every=FLAGS.print_every, use_resize=FLAGS.use_resize, 
                                    eps=FLAGS.eps, max_eval=FLAGS.max_evals,
                                    over=FLAGS.over, rounding=FLAGS.rounding,
                                    max_f=FLAGS.max_f, subspace_attack=FLAGS.subspace_attack,
                                    subspace_dim=FLAGS.subspace_dimension)

            result = attack.attack_batch(inputs, targets)
        elif FLAGS.attack=='combi':
            result = ParsimoniousAttack(loss_func, inputs, 
                                         np.argmax(targets[0]), FLAGS)
        elif FLAGS.attack=='square':
            result = square_attack_linf(model=model, x=inputs, 
                                    y=np.array(targets[0], dtype=int), 
                                    eps = FLAGS.eps, n_iters=FLAGS.max_evals, 
                                    p_init=FLAGS.p_init, targeted=True, 
                                    loss_type='cross_entropy', 
                                    print_every=FLAGS.print_every,subspace_attack=FLAGS.subspace_attack,
                                    subspace_dim=FLAGS.subspace_dimension)
                                    

        adv, eval_costs, summary, Success = result
        
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