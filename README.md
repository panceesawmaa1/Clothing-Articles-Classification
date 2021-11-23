# Clothing Articles Classification

### Table of contents

* [Description](#Description)
  
* [Minimum Requirements](#Minimum Requirements)
  
* [Setup Instructions](#Setup Instructions)

### Description

This repository contains the appropriate code for a simple Clothing Articles
classifiers. Implemented models are trained with ```fashion-mnist``` dataset.

Loss function used:

* ```categorical_crossentropy```

Metric used:

* ```accuracy```

Models implemented:

* ```BaseLineClassifier``` a simple basline CNN classifier, consisting of 7 layers,
  ```Conv2D```, ```MaxPool2D```, ```Flatten``` and ```Dense``` layers. This model obtained **91.6%** testing accuracy
  after training with 30 epochs.


* ```Classifier``` a simple CNN classifier with ```Dropout``` regularization, consisting of 9 layers,
  ```Conv2D```, ```MaxPool2D```, ```Flatten```, ```Dropout``` and ```Dense``` layers.
  This model obtained **92.5%** testing accuracy after training with 30 epochs.
  

* ```DeeperClassifier``` a deeper CNN classifier with ```Dropout``` and ```l2``` 
  regularizations, consisting of 24 layers,
  ```Conv2D```, ```MaxPool2D```, ```BatchNormalization```, ```Flatten```,
  ```Dropout``` and ```Dense``` layers.
  This model obtained **93.15%** testing accuracy after training with 30 epochs.
  
### Minimum Requirements

* Python >= 3.6

### Setup Instructions

* Download Fashion-MNIST dataset by running the following commands:
  ```shell
  $ cd scripts/
  $ bash download_data.sh
  ```

* Then, set up project environment by running the following command:
  ```shell
  $ bash setup_environment.sh
  ```
* To run the model, run the following commands:
  ```shell
  $ cd ..
  $ python examples/run.py
  ```
  
