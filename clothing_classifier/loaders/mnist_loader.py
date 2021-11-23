"""
MNIST Dataset Loader Module
---------------------------

This loader was implemented by the help of the MNIST authors reader,
for more info see
<https://github.com/zalandoresearch/fashion-mnist/blob/master/utils/mnist_reader.py>.

"""
import os
import gzip
import numpy as np


class MNISTLoader:
    """
    MNISTReader class
    Loads train or test MNIST data as specified by the user.
    """

    def __init__(self, path: str = 'data/fashion') -> None:
        self.path = path

    def set_data_path(self, path: str) -> None:
        """
        This function sets the reader's data path.
        :param path: str
            Data path
        :return: None
        """
        self.path = path

    def load_train_data(self):
        """
        This function loads the training images and labels.
        :return: Training images and labels.
        """
        kind = 'train'
        return self.__load(kind)

    def load_test_data(self):
        """
        This function loads the testing images and labels.
        :return: Testing images and labels.
        """
        kind = 't10k'
        return self.__load(kind)

    def __load(self, kind: str):
        """
        Function to load train or test MNIST data from the path specified.

        :param kind: str
            Either 'train' to load train data or 'test' to load testing data.

        :return Training or testing images and labels.
        """
        labels_path = os.path.join(self.path,
                                   '%s-labels-idx1-ubyte.gz'
                                   % kind)

        images_path = os.path.join(self.path,
                                   '%s-images-idx3-ubyte.gz'
                                   % kind)

        with gzip.open(labels_path, 'rb') as lbpath:
            labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                                   offset=8)

        with gzip.open(images_path, 'rb') as imgpath:
            images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                                   offset=16).reshape(len(labels), 784)

        return images, labels
