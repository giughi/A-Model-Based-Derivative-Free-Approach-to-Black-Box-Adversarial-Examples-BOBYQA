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
CHECKPOINT_DIR=./Data/Imagenet
mkdir ${CHECKPOINT_DIR}
wget http://jaina.cs.ucdavis.edu/datasets/adv/imagenet/img.tar.gz
tar -xvf img.tar.gz -C ./Data/Imagenet
rm img.tar.gz
```
however, if it fails to open the tar file, plese export all of the images in the folder ./Data/Imagenet/images
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

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
