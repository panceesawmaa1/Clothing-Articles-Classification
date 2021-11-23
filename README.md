# Clothing Articles Classification

### Table of contents

- [Description](#Description)

- [Minimum Requirements](#Minimum-Requirements)

- [Setup Instructions](#Setup-Instructions)

### Description

This repository contains the appropriate code for a simple Clothing Articles classifiers. Implemented models are trained
with ```fashion-mnist``` dataset.

Loss function used:

* ```categorical_crossentropy```

Metric used:

* ```accuracy```
 
  

Models implemented:

### 1. ```BaseLineClassifier```
A simple basline CNN classifier, consisting of 7 layers,
```Conv2D```, ```MaxPool2D```, ```Flatten``` and ```Dense``` layers. This model obtained **91.6%** testing accuracy
after training with 30 epochs.

  Model Summary:

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 28, 28, 32)        832       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 14, 14, 32)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 14, 14, 64)        51264     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 7, 7, 64)          0         
_________________________________________________________________
flatten (Flatten)            (None, 3136)              0         
_________________________________________________________________
dense (Dense)                (None, 1024)              3212288   
_________________________________________________________________
output_layer (Dense)         (None, 10)                10250     
=================================================================
Total params: 3,274,634
Trainable params: 3,274,634
Non-trainable params: 0
_________________________________________________________________
```

**FLOPs Report:**

**NOTE:**

>Convolutions - FLOPs = 2x Number of Kernel x Kernel Shape x Output Shape

>Fully Connected Layers - FLOPs = 2x Input Size x Output Size

>1 MAC = 2 FLOPs

* Layer ```conv2d``` FLOps = 2 * 32 * (5 * 5) * (28 * 28) = 1,254,400 FLOPs
* Layer ```conv2d_1``` FLOPs = 2 * 64 * (5 * 5 * 64) * (14 * 14) = 40,140,800 FLOPs
* Layer ```dense``` FLOPs = 2 * 3136 * 1024 = 6,422,528 FLOPs
* Layer ```output_layer``` FLOPs = 2 * 1024 * 10 = 20,480 FLOPs

**This model performs 47,838,208 FLOPs (~46 MFLOPs), 23,919,104 MACs (~23 Mega-MACs) 
for ```convolution``` and ```fully-connected``` layers only.**



### 2. ```Classifier```
A simple CNN classifier with ```Dropout``` regularization, consisting of 9 layers,
```Conv2D```, ```MaxPool2D```, ```Flatten```, ```Dropout``` and ```Dense``` layers. This model obtained **92.5%**
testing accuracy after training with 30 epochs.

  Model Summary:

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 28, 28, 32)        832       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 14, 14, 32)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 14, 14, 64)        51264     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 7, 7, 64)          0         
_________________________________________________________________
flatten (Flatten)            (None, 3136)              0         
_________________________________________________________________
dropout (Dropout)            (None, 3136)              0         
_________________________________________________________________
dense (Dense)                (None, 1024)              3212288   
_________________________________________________________________
dropout_1 (Dropout)          (None, 1024)              0         
_________________________________________________________________
output_layer (Dense)         (None, 10)                10250     
=================================================================
Total params: 3,274,634
Trainable params: 3,274,634
Non-trainable params: 0
_________________________________________________________________

```

**FLOPs Report:**

**NOTE:**

>Convolutions - FLOPs = 2x Number of Kernel x Kernel Shape x Output Shape

>Fully Connected Layers - FLOPs = 2x Input Size x Output Size

>1 MAC = 2 FLOPs

* Layer ```conv2d``` FLOps = 2 * 32 * (5 * 5) * (28 * 28) = 1,254,400 FLOPs
* Layer ```conv2d_1``` FLOPs = 2 * 64 * (5 * 5 * 64) * (14 * 14) = 40,140,800 FLOPs
* Layer ```dense``` FLOPs = 2 * 3136 * 1024 = 6,422,528 FLOPs
* Layer ```output_layer``` FLOPs = 2 * 1024 * 10 = 20,480 FLOPs

**This model performs 47,838,208 FLOPs (~46 MFLOPs), 23,919,104 MACs (~23 Mega-MACs)
for ```convolution``` and ```fully-connected``` layers only.**

### 3. ```DeeperClassifier```
   A deeper CNN classifier with ```Dropout``` and ```l2```
  regularizations, consisting of 24 layers,
  ```Conv2D```, ```MaxPool2D```, ```BatchNormalization```, ```Flatten```,
  ```Dropout``` and ```Dense``` layers. This model obtained **93.15%** testing accuracy after training with 30 epochs.

  Model Summary:

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 28, 28, 64)        640       
_________________________________________________________________
batch_normalization (BatchNo (None, 28, 28, 64)        256       
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 26, 26, 64)        36928     
_________________________________________________________________
batch_normalization_1 (Batch (None, 26, 26, 64)        256       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 13, 13, 64)        0         
_________________________________________________________________
dropout (Dropout)            (None, 13, 13, 64)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 13, 13, 128)       73856     
_________________________________________________________________
batch_normalization_2 (Batch (None, 13, 13, 128)       512       
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 11, 11, 128)       147584    
_________________________________________________________________
batch_normalization_3 (Batch (None, 11, 11, 128)       512       
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 5, 5, 128)         0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 5, 5, 128)         0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 5, 5, 256)         295168    
_________________________________________________________________
batch_normalization_4 (Batch (None, 5, 5, 256)         1024      
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 3, 3, 256)         590080    
_________________________________________________________________
batch_normalization_5 (Batch (None, 3, 3, 256)         1024      
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 1, 1, 256)         0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 1, 1, 256)         0         
_________________________________________________________________
flatten (Flatten)            (None, 256)               0         
_________________________________________________________________
dense (Dense)                (None, 1024)              263168    
_________________________________________________________________
dropout_3 (Dropout)          (None, 1024)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 512)               524800    
_________________________________________________________________
dropout_4 (Dropout)          (None, 512)               0         
_________________________________________________________________
output_layer (Dense)         (None, 10)                5130      
=================================================================
Total params: 1,940,938
Trainable params: 1,939,146
Non-trainable params: 1,792
_________________________________________________________________
```

**FLOPs Report:**

**NOTE:**

>Convolutions - FLOPs = 2x Number of Kernel x Kernel Shape x Output Shape

>Fully Connected Layers - FLOPs = 2x Input Size x Output Size

>1 MAC = 2 FLOPs

* Layer ```conv2d``` FLOps = 2 * 64 * (3 * 3) * (28 * 28) = 903,168 FLOPs
* Layer ```conv2d_1``` FLOPs = 2 * 64 * (3 * 3 * 64) * (26 * 26) = 49,840,128 FLOPs
* Layer ```conv2d_2``` FLOPs = 2 * 128 * (3 * 3 * 128) * (13 * 13) = 49,840,128 FLOPs
* Layer ```conv2d_3``` FLOPs = 2 * 128 * (3 * 3 * 128) * (11 * 11) = 35,684,352 FLOPs
* Layer ```conv2d_4``` FLOPs = 2 * 256 * (3 * 3 * 256) * (5 * 5) = 29,491,200 FLOPs
* Layer ```conv2d_5``` FLOPs = 2 * 256 * (3 * 3 * 256) * (3 * 3) = 10,616,832 FLOPs
* Layer ```dense``` FLOPs = 2 * 256 * 1024 = 524,288 FLOPs
* Layer ```dense_1``` FLOPs = 2 * 1024 * 512 = 1,048,576 FLOPs
* Layer ```output_layer``` FLOPs = 2 * 512 * 10 = 10,240 FLOPs

**This model performs 177,958,912 FLOPs (~170 MFLOPs), 88,979,456 MACs (~85 Mega-MACs)
for ```convolution``` and ```fully-connected``` layers only.**

#### Conclusion

* Increasing the model size represented in ```DeeperClassifier``` made the model
perform **3.6X FLOPs** (for ```convolution``` and ```fully-connected``` layers only)
more than model represented in ```Classifier``` (which is less deep) 
where ```DeeperClassifier``` performs **+0.65%** better than ```Classifier```
in terms of testing accuracy. We can go with the simpler model to decrease 
number of FLOPs performed by the model, especially that the accuracy enhancement 
  is not that much significant.
 
* The ```convolution``` layers are more expensive than ```fully-connected``` ones
from this experiment.



### Minimum Requirements

* Python >= 3.6

### Setup Instructions

* Clone the repository.

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
  
