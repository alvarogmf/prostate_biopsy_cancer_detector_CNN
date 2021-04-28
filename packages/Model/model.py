import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras_preprocessing.image import ImageDataGenerator



def model(train_generator, validation_generator):
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

    print('starting model...')
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(258, 258, 3)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    model.compile(optimizer='adam',
                  loss="binary_crossentropy",
                  metrics=['accuracy'])
    print('training model... this will take some time, you can go grab a coffe or watch a movie ;)')
    model.fit(
        train_generator,
        validation_data=validation_generator,
        batch_size=16,
        epochs=50)

    return model


def save_model_h5(model, model_path="model/", model_name="model"):
    """
    Saves model to a .h5 file
    :param model: Model to be saved as a .h5 file in the specified path
    :param model_path: Path where the .h5 file model will be saved to. Default: ../data/model
    :param model_name: Model's name. Default: model
    :return: Saves the model to the specified path under the specified name
    """
    model.save(os.path.join(model_path, f"{model_name}.h5"))
    print(f"Saved model to {model_path}/{model_name}")


def load_model_h5(model_path="model/", model_name="model"):
    """
    Loads a previously saved model in a .h5 file
    :param model_path: Path where the .h5 file model will be read from. Default: ../data/model
    :param model_name: Model's name. Default: model
    :return: Loaded model
    """
    try:
        # Loading H5 file
        loaded_model = load_model(os.path.join(model_path, f"{model_name}.h5"))
        print(f"Model loaded successfully -> {model_name}.h5")
        return loaded_model
    except Exception as e:
        print("Model couldn't be loaded")
        exit()

