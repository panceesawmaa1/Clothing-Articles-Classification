from clothing_classifier.loaders.mnist_loader import MNISTLoader
from clothing_classifier.models.basline_classifier import BaseLineClassifier
from clothing_classifier.models.classifier import Classifier
from clothing_classifier.models.deeper_classifier import DeeperClassifier

from tensorflow.keras.optimizers import *
from tensorflow.keras.utils import to_categorical


def main():
    # Create data loader object
    loader = MNISTLoader()

    # Get train and test data with the help of the data loader
    train_x, train_y = loader.load_train_data()
    test_x, test_y = loader.load_test_data()

    # Reshape train and test input images to be 28x28x1
    train_x = train_x.reshape([-1, 28, 28, 1])
    test_x = test_x.reshape([-1, 28, 28, 1])

    # Normalize train and test input images
    train_x = train_x / 255
    test_x = test_x / 255

    print("Train Data Shape: ", train_x.shape, "Train Labels Shape: ", train_y.shape)
    print("Test Data Shape: ", test_x.shape, "Test Labels Shape: ", test_y.shape)

    # Apply One Hot Encoding to train and test labels
    train_y = to_categorical(train_y)
    test_y = to_categorical(test_y)

    # Model training parameters
    epochs = 30
    valid_split = .2

    # Model creation, of type BaseLineClassifier
    # model = BaseLineClassifier()
    model = Classifier()
    # model = DeeperClassifier()

    # Register losses
    model.register_losses(['categorical_crossentropy'])

    # Register metrics
    model.register_metrics(['accuracy'])

    # Model building
    model.build()

    # Model compilation
    learning_rate = 1e-4
    optimizer = Adam(learning_rate)
    model.compile(
        optimizer=optimizer
    )

    # Print model summary
    model.summary()

    # Model training
    model.train(
        train_x,
        train_y,
        validation_split=valid_split,
        epochs=epochs
    )

    # Model evaluation
    loss, accuracy = model.evaluate(test_x, test_y)

    print()
    print("Model Evaluation Results")
    print("========================")
    print("Model loss : ", loss)
    print("Model accuracy : ", accuracy)


if __name__ == '__main__':
    main()
