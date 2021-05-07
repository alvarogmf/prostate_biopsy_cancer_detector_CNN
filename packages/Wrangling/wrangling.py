import pandas as pd
import numpy as np
from keras_preprocessing.image import ImageDataGenerator


def create_clean_csv(original_csv, format = '.png'):
    """
    Takes the original Dataframe and creates a simplified Dataframe with Cancer or no cancer.
    :param original_csv: Original Dataframe
    :param format: The format in which the images will be used to train the model
    :return: New Dataframe
    """
    path_input_csv['path'] = path_input_csv['image_id'] + format
    train_csv['binary_score'] = np.where((train_csv['isup_grade'] < 1), 'no cancer','cancer')

    df_clean = pd.DataFrame(path_input_csv[['path', 'binary_score']])
    return df_clean


def image_generator(dataframe,directory, target_size = (258, 258), batch_size = 64):
    """
    Divides the images into two groups, one for training and one for validation and transforms them into something trainable.
    :param dataframe: Dataframe to take the information
    :param directory: Folder where the files are stored
    :param target_size: Target size of the image. Default: (258, 258)
    :param batch_size: batch size. Default: 64
    :return: two datasets objects, train_generator and validation_generator
    """
    datagen = ImageDataGenerator(rescale=1. / 255., validation_split=0.2)
    train_generator = datagen.flow_from_dataframe(
        dataframe=dataframe,
        directory=directory,
        x_col="path",
        y_col="binary_score",
        target_size=target_size,
        color_mode="rgb",
        batch_size=batch_size,
        save_to_dir='png_images_resized/',
        save_format="png",
        class_mode="binary",
        subset='training',
        shuffle=True
    )

    validation_generator = datagen.flow_from_dataframe(
        dataframe=dataframe,
        directory=directory,
        x_col="path",
        y_col="binary_score",
        target_size=target_size,
        color_mode="rgb",
        batch_size=batch_size,
        save_to_dir='png_images_resized/',
        save_format="png",
        class_mode="binary",
        subset='validation',
        shuffle=True
    )
    return train_generator, validation_generator
