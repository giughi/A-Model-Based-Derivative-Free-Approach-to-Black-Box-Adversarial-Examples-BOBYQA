import os
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

import Attack_Code.GenAttack.utils as utils
from Attack_Code.GenAttack.genattack_tf2_PyTorch import GenAttack2
from Attack_Code.MNIST_CIFAR_boby_gen import BlackBox

dir_path = os.path.dirname(os.path.realpath(__file__))
main_dir = os.path.abspath(os.path.join(dir_path, os.pardir))

from PIL import Image
import torch as ch
from robustness.datasets import CIFAR as CIFAR_robustness
from robustness.model_utils import make_and_restore_model

flags = tf.app.flags
flags.DEFINE_string('dataset', 'cifar10', 'model name')
flags.DEFINE_integer('test_size', 1, 'Number of test images.')
flags.DEFINE_integer('max_evals', 3000, 'Maximum number of function evaluations.')
flags.DEFINE_integer('print_every', 10, 'Every iterations the attack function has to print out.')
flags.DEFINE_integer('seed', 1216, 'random seed')
flags.DEFINE_bool('Adversary_trained', False, ' Use the adversarially trained nets')
flags.DEFINE_string('description', '', 'Description for how to save the results')
flags.DEFINE_string('attack_type', '', 'attack used ')
flags.DEFINE_float('eps', 0.1, 'perturbation energy')
flags.DEFINE_bool('internal_summaries', False, 'perturbation energy')

flags.DEFINE_float('mutation_rate', 0.005, 'Mutation rate')
flags.DEFINE_float('alpha', 0.20, 'Step size')
flags.DEFINE_integer('pop_size', 6, 'Population size')
flags.DEFINE_integer('resize_dim', 32, 'Reduced dimension for dimensionality reduction')
flags.DEFINE_bool('adaptive', True, 'Turns on the dynamic scaling of mutation prameters')
flags.DEFINE_string('model', 'inception', 'model name')
flags.DEFINE_integer('target', None, 'target class. if not provided will be random')

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
        logit,_ =self.model(img) 
        return logit.detach().numpy()

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

class args_CLASS(object): #= vars(parser.parse_args())
    dataset = 'cifar10'            # choices=["mnist", "cifar10", "imagenet"], default="mnist")
    save = './Results/'
    numimg = 0                  # type=int, default=0, help = "number of test images to attack")
    maxiter = 0                  # type=int, default=0, help = "set 0 to use default value")
    print_every = 100                # type=int, default=100, help = "print objs every PRINT_EVERY iterations")
    early_stop_iters = 100   # type=int, default=100, help = "print objs every EARLY_STOP_ITER iterations, 0 is maxiter//10")
    firstimg = 0                  # type=int, default=0)
    untargeted = True               # action='store_true')
    use_resize = False              # action='store_true', help = "resize image (only works on imagenet!)")
    seed = 1216               # type=int, default=1216)


if __name__ == '__main__':
    args = args_CLASS()
    args.dataset = FLAGS.dataset

    if args.dataset == 'mnist':
        args.save = './Results/MNIST'
    elif args.dataset == 'cifar10':
        args.save = './Results/CIFAR'
    if args.maxiter == 0:
        if args.dataset == "mnist":
            args.maxiter = 3000
        else:
            args.maxiter = 3000
    
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    args_pd = {}
    for elem in dir(args):
        if elem[:2] != '__':
            args_pd[elem] = eval('args.'+elem)

    args = args_pd

    args['use_resize'] = False
    args['attack'] = 'BOBYQA'

    args['print_every'] = FLAGS.print_every
    args['max_iter'] = FLAGS.max_evals
    args['untargeted'] = False

    BATCH = 50
    use_log = False
    vector_eval = []
    ORDERED_DOMAIN = True
    img_dist = False
    mix_dist = True
    MAX_EVAL = FLAGS.max_evals
    q = BATCH

    global_summary = []

    L_inf_var = FLAGS.eps

    list_boby = []
    list_gene = []
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=4,
                                        inter_op_parallelism_threads=4)


    # load network
    print('Loading model', args['dataset'])
    if args['dataset'] == "mnist":
        if FLAGS.Adversary_trained:
            data, model = MNIST(), MNISTModel(main_dir + "/Models/mnist-distilled-100", sess, use_log)
        else:
            data, model = MNIST(), MNISTModel(main_dir + "/Models/mnist", sess, use_log)
    elif args['dataset'] == "cifar10":
        if FLAGS.Adversary_trained:
            data, model = CIFAR(), CIFARModel(main_dir + "/Models/cifar-distilled-100", sess, use_log)
        else:
            data = CIFAR()

    ds = CIFAR_robustness('./Data/CIFAR-10')

    model,_ = make_and_restore_model(arch='resnet50', dataset=ds, resume_path='./Models/cifar_robust.pt',
                                    parallel=False)
    model.eval()
    model = Model_Class(model.float())


    print('Done...')

    args['numimg'] = FLAGS.test_size
    print('Using', args['numimg'], 'test images with ', L_inf_var,' energy')
            

    attack_gene = GenAttack2(model=model,
                    pop_size=FLAGS.pop_size,
                    mutation_rate = FLAGS.mutation_rate,
                    eps=FLAGS.eps,
                    max_steps=int(FLAGS.max_evals/(FLAGS.pop_size-1)),
                    alpha=FLAGS.alpha,
                    resize_dim=FLAGS.resize_dim,
                    adaptive=FLAGS.adaptive,
                    num_classes=10)
    
    # BlackBox(sess, model, batch_size=6, max_iterations=args['maxiter'], print_every=args['print_every'],
    #                 early_stop_iters=args['early_stop_iters'], confidence=0, targeted=not args['untargeted'], 
    #                 use_resize=args['use_resize'],L_inf=L_inf_var,
    #                 GenAttack=True,max_eval = MAX_EVAL,q = q)

    print('Generate data')
    all_inputs, all_targets, all_labels, all_true_ids = generate_data(data, samples=args['numimg'], targeted=not args['untargeted'],
                                    start=args['firstimg'])
    print('Done...')
    
    img_no = 0
    total_success = 0
    
    if FLAGS.dataset == 'mnist':
        saving_dir = main_dir + '/Results/MNIST/' + FLAGS.description
    else:
        saving_dir = main_dir + '/Results/CIFAR/' + FLAGS.description
    

    case ='_madry'

    saving_name_gene = saving_dir+'gene_L_inf_'+str(L_inf_var)+'_max_eval_'+ str(FLAGS.max_evals) + case + '.txt'
    
    already_done_init = 0
    already_done = 0

    if os.path.exists(saving_name_gene):
        if os.path.getsize(saving_name_gene)>0:
            with open(saving_name_gene, "rb") as fp:
                list_gene = pickle.load(fp)
            already_done_init = len(list_gene)
    already_done = already_done_init

    if already_done_init>0:
        found = False
        while not found:
            lab = list_boby[-1][2]
            tar = list_boby[-1][3] 
            # if FLAGS.attack_type == 'gene':    
            #     lab = list_gene[-1][2]
            #     tar = list_gene[-1][3] 
            
            lab_ = np.argmax(all_labels[already_done_init-1:already_done_init])
            tar_ = np.argmax(all_targets[already_done_init-1:already_done_init])
            print(already_done, lab, tar, lab_, tar_)
            already_done_init +=1
            if lab == lab_ and tar==tar_:
                found = True
                already_done_init -= 1

    print('We have done ', already_done, ' attacks but are at iteration ', already_done_init)
    
    for i in range(already_done_init,args['numimg']):
        inputs = all_inputs[i:i+1]
        targets = all_targets[i:i+1]
        labels = all_labels[i:i+1]
        print("true labels:", np.argmax(labels), labels)
        print("target:", np.argmax(targets), targets)
        # test if the image is correctly classified
        original_predict = model.predict(inputs)
        original_predict = np.squeeze(original_predict)
        original_prob = np.sort(original_predict)
        original_class = np.argsort(original_predict)
        print("original probabilities:", original_prob[-1:-6:-1])
        print("original classification:", original_class[-1:-6:-1])
        print("original probabilities (most unlikely):", original_prob[:6])
        print("original classification (most unlikely):", original_class[:6])
        if original_class[-1] != np.argmax(labels):
            print("skip wrongly classified image no. {}, original class {}, classified as {}".format(i, np.argmax(labels), original_class[-1]))
            continue

        img_no += 1
        timestart = time.time()
        # adversarial generation
        success, adv_gene, eval_costs_gene = attack_gene.attack(inputs, np.argmax(targets))

        if len(adv_gene.shape) == 3:
            adv_gene = adv_gene.reshape((1,) + adv_gene.shape)

        timeend = time.time()
        if success:
            adversarial_class = np.argmax(targets[0])
            adversarial_predict_gene = model.predict(adv_gene)
            print('#### FINAL ##### pred', np.argmax(adversarial_predict_gene), eval_costs_gene)
            adversarial_predict_gene = np.squeeze(adversarial_predict_gene)
            adversarial_prob_gene = np.sort(adversarial_predict_gene)
            adversarial_class_gene = np.argsort(adversarial_predict_gene)

            list_gene.append([eval_costs_gene, original_prob[-1]-adversarial_prob_gene[-1],
                            np.argmax(labels), np.argmax(targets)])
        else:
            list_gene.append([eval_costs_gene, -1,
                            np.argmax(labels), np.argmax(targets)])
        print("[STATS][L1] total = {}, seq = {}, batch = {}prev_class = {}, new_class = {}".format(img_no, i, BATCH, original_class[-1], adversarial_class))
        sys.stdout.flush()

        # if FLAGS.internal_summaries:
        #     global_summary.append(summary)
        #     with open(saving_dir+'summary_gene_L_inf_'+str(L_inf_var)+'_max_eval_'+ str(FLAGS.max_evals) +'_batch_'+str(BATCH)+'.txt', "wb") as fp:
        #         pickle.dump(global_summary, fp)

        with open(saving_name_gene , "wb") as fp:
                pickle.dump(list_gene, fp)

        print('saved')