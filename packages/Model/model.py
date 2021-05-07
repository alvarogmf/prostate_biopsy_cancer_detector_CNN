import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras_preprocessing.image import ImageDataGenerator
import os


def model(train_generator, validation_generator):
    """
    Function to train the model. NOTE: this model requires GPU, so tensorflow-gpu needs to be installed.
    :param train_generator: the images to be used to train the model.
    :param validation_generator: the images to be used to test the model.
    :return: the model
    """
    # Activating the GPU and limiting its capacity to 4Gb, change memory_limit if needed.
    print('setting GPU configuration...')
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 4GB of memory on the first GPU
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

    # Different layers of the Model
    print('starting model...')
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(256, 256, 3)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(64, activation="relu"))

    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Dense(32, activation="relu"))

    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Dense(16, activation="relu"))

    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    # Model compiler
    model.compile(optimizer='adam',
                  loss="binary_crossentropy",
                  metrics=['accuracy'])
    print('training model... this will take some time, you can go grab a coffe or watch a movie ;)')

    # Model fit
    model.fit(
        train_generator,
        validation_data=validation_generator,
        batch_size=32,
        epochs=50)

    return model


def save_model_h5(model, model_path="model/", model_name="model"):
    """
    Saves model to a .h5 file
    :param model: Model to be saved as a .h5 file in the specified path
    :param model_path: Path where the .h5 file model will be saved to. Default: model/
    :param model_name: Model's name. Default: model
    :return: Saves the model to the specified path under the specified name
    """
    model.save(os.path.join(model_path, f"{model_name}.h5"))
    print(f"Saved model to {model_path}/{model_name}")


def load_model_h5(model_path="model/", model_name="model"):
    """
    Loads a previously saved model in a .h5 file
    :param model_path: Path where the .h5 file model will be read from. Default: model/
    :param model_name: Model's name. Default: model
    :return: Loaded model
    """
    try:
        # Loading H5 file
        loaded_model = tf.keras.models.load_model(os.path.join(model_path, f"{model_name}.h5"))
        print(f"Model loaded successfully -> {model_name}.h5")
        return loaded_model
    except Exception as e:
        print("Model couldn't be loaded")
        exit()


