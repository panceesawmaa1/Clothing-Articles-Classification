"""
Base Model Module.
"""

from abc import ABC, abstractmethod


class BaseModel(ABC):
    """
    BaseModel class
    This class is an abstract model interface, different classifiers inherits and
    implements its methods.
    """

    def __init__(self):
        self.callbacks = []
        self.metrics = []
        self.losses = []

    def register_callbacks(self, callbacks) -> None:
        """
        This functions registers model callbacks.
        :param callbacks: list
        :return: None
        """
        self.callbacks = callbacks

    def register_metrics(self, metrics) -> None:
        """
        This functions registers model metrics.
        :param metrics: list
        :return: None
        """
        self.metrics = metrics

    def register_losses(self, losses) -> None:
        """
        This functions registers model losses.
        :param losses: list
        :return: None
        """
        self.losses = losses

    @abstractmethod
    def build(self) -> None:
        """
        This function builds the model.
        :return: None
        """

    @abstractmethod
    def compile(self, optimizer) -> None:
        """
        This function compiles the model.
        :param optimizer:
        :return: None
        """

    @abstractmethod
    def train(self, train_x, train_y, validation_split, epochs) -> None:
        """
        This function trains the model with train and validation data, and with
        the specified number of epochs.
        :param self:
        :param train_x:
        :param train_y:
        :param validation_split:
        :param epochs: int
        :return: None
        """

    @abstractmethod
    def evaluate(self, test_x, test_y):
        """
        This function tests/evaluates the model on testing data.
        :param test_x:
        :param test_y:
        :return:
        """

    @abstractmethod
    def infer(self, x_data) -> None:
        """
        This function generates model predictions.
        :param x_data:
        :return:
        """

    @abstractmethod
    def summary(self) -> None:
        """
        This function prints the model summary.
        :return: None
        """
