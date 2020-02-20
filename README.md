# A-Model-Based-Derivative-Free-Approach-to-Black-Box-Adversarial-Examples-BOBYQA
Scripts that allow the reproduction of the results presented in the article "A Model-Based Derivative-Free Approach to Black-Box Adversarial Examples: BOBYQA".


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

```
pip install -r requirements.txt 
```

### Installing

It is first required to download/train the different datasets and models that are going to be attacked.

For MNIST and CIFAR it is necessary to run the following comand
```
python Setups/Data_and_Model/train_CIFAR_MNIST_models.py
```
To download the ImageNet dataset
```
CHECKPOINT_DIR=./Data/ImageNet
mkdir ${CHECKPOINT_DIR}
wget http://jaina.cs.ucdavis.edu/datasets/adv/imagenet/img.tar.gz
tar -xvf img.tar.gz -C ./Data/ImageNet
rm img.tar.gz
mv ./Data/ImageNet/imgs ./Data/ImageNet/images
rm ./Data/ImageNet/imgs

```
and to install the inception-v3 net 
```
python Setups/Data_and_Model/setup_inception.py
```


For Inception Adversarially Trained Nets run

```
CHECKPOINT_DIR=./Models
mkdir ${CHECKPOINT_DIR}
wget http://download.tensorflow.org/models/ens_adv_inception_resnet_v2_2017_08_18.tar.gz
tar -xvf ens_adv_inception_resnet_v2_2017_08_18.tar.gz
mv ens_adv_inception_resnet_v2.ckpt* ${CHECKPOINT_DIR}
rm ens_adv_inception_resnet_v2_2017_08_18.tar.gz
```


End with an example of getting some data out of the system or using it for a little demo

## Running the tests

To run the experiments suggested in the manuscript, it is necessary to run separatly the following codes for each dataset.

These datasets are treated separately because of how the net net is stored and how each net then requires a slightly different implementation.

### MNIST

The normally trained net is attacked with:

```
# COMBI 
python Setups/MNIST_CIFAR_COMBI.py --epsilon=0.15 --dim_image=28 --num_channels=1 --dataset=mnist
# GENE
python Setups/MNIST_CIFAR_GENE.py --eps=0.15 --test_size=10 --dataset=mnist --Adversary_trained=False
# SQUARE
python Setups/MNIST_CIFAR_SQUARE.py --eps=0.15 --test_size=10 --dataset=mnist --Adversary_trained=False
# BOBYQA
python Setups/MNIST_CIFAR_BOBYQA.py --eps=0.15 --test_size=10 --dataset=mnist --Adversary_trained=False
```

The Adversarially trained net instead with 
```
# COMBI
python Setups/MNIST_CIFAR_COMBI.py --epsilon=0.15 --dim_image=28 --num_channels=1 --dataset=mnist --Adversary_trained=True
# GENE
python Setups/MNIST_CIFAR_GENE.py --eps=0.15 --test_size=10 --dataset=mnist --Adversary_trained=True
# SQUARE
python Setups/MNIST_CIFAR_SQUARE.py --eps=0.15 --test_size=10 --dataset=mnist --Adversary_trained=True
# BOBYQA 
python Setups/MNIST_CIFAR_BOBYQA.py --eps=0.15 --test_size=10 --dataset=mnist --Adversary_trained=True
```

### CIFAR

The normally trained net is attacked with

```
# COMBI
python Setups/MNIST_CIFAR_COMBI.py --epsilon=0.15 --dim_image=32 --num_channels=3 --dataset=cifar10
# GENE
python Setups/MNIST_CIFAR_GENE.py --eps=0.15 --test_size=10 --dataset=cifar10 --Adversary_trained=False
# SQUARE
python Setups/MNIST_CIFAR_SQUARE.py --eps=0.15 --test_size=10 --dataset=cifar10 --Adversary_trained=False
# BOBYQA
python Setups/MNIST_CIFAR_BOBYQA.py --eps=0.15 --test_size=10 --dataset=cifar10 --Adversary_trained=False
```

While the adversarially trained net was attacked with:

```
# COMBI
python Setups/MNIST_CIFAR_COMBI.py --epsilon=0.15 --dim_image=32 --num_channels=3 --dataset=cifar10 --Adversary_trained=True
# GENE
python Setups/MNIST_CIFAR_BOBYQA.py --eps=0.15 --test_size=10 --dataset=cifar10 --Adversary_trained=True
# SQUARE
python Setups/MNIST_CIFAR_SQUARE.py --eps=0.15 --test_size=10 --dataset=cifar10 --Adversary_trained=False
# BOBYQA
python Setups/MNIST_CIFAR_BOBYQA.py --eps=0.15 --test_size=10 --dataset=cifar10 --Adversary_trained=True
```

### ImageNet

In the normal case   

```
# COMBI

# GENE

# SQUARE

# BOBYQA

```

While in the adversary case

```
# COMBI

# GENE

# SQUARE

# BOBYQA

```

## Analysis of the Results

To analyse the results it is possible to use the following functions. Though, to run them it is necessary to write inside of the script what energy bounds have been considered.


