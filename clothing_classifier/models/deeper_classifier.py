"""
Classifier Module.
"""

from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2

from clothing_classifier.models.base_model import BaseModel


class DeeperClassifier(BaseModel):
    """
    DeeperClassifier class.
    This class inherits from BaseModel and implements its methods.
    The model architecture can be described as a 9-layer CNN.
    """

    def __init__(self):
        super().__init__()
        self.accuracy = 0
        self.loss = 0

    def register_callbacks(self, callbacks) -> None:
        super().register_callbacks(callbacks)

    def register_metrics(self, metrics) -> None:
        super().register_metrics(metrics)

    def register_losses(self, losses) -> None:
        super().register_losses(losses)

    def build(self) -> None:
        l_2 = l2(0.001)
        self.model: Sequential = Sequential([
            Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=l_2,
                   input_shape=[28, 28, 1], name='input_layer'),
            BatchNormalization(),

            Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_regularizer=l_2),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.20),

            Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=l_2),
            BatchNormalization(),

            Conv2D(128, kernel_size=(3, 3), activation='relu', kernel_regularizer=l_2),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.30),

            Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=l_2),
            BatchNormalization(),

            Conv2D(256, kernel_size=(3, 3), activation='relu', kernel_regularizer=l_2),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.4),

            Flatten(),

            Dense(1024, activation='relu', kernel_regularizer=l_2),
            Dropout(0.3),

            Dense(512, activation='relu', kernel_regularizer=l_2),
            Dropout(0.2),

            Dense(10, activation='softmax', name='output_layer')
        ])

    def compile(self, optimizer) -> None:
        self.model.compile(
            optimizer=optimizer, loss=self.losses, metrics=self.metrics
        )

    def train(self, train_x, train_y, validation_split, epochs) -> None:
        self.model.fit(
            train_x,
            train_y,
            validation_split=validation_split,
            batch_size=64,
            epochs=epochs,
            verbose=2
        )

    def evaluate(self, test_x, test_y):
        self.loss, self.accuracy = self.model.evaluate(test_x, test_y, verbose=0)

        return self.loss, self.accuracy

    def infer(self, x_data) -> None:
        pass

    def summary(self) -> None:
        self.model.summary()
