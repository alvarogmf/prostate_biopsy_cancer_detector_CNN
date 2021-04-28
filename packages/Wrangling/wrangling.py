import pandas as pd
import numpy as np
from keras_preprocessing.image import ImageDataGenerator


def create_clean_csv(original_csv, format = '.png'):
    path_input_csv['path'] = path_input_csv['image_id'] + format
    path_input_csv['binary_score'] = np.where(
        (path_input_csv['data_provider'] == 'karolinska') & (path_input_csv['isup_grade'] < 3), 'benigno',
        np.where((path_input_csv['data_provider'] == 'radboud') & (path_input_csv['isup_grade'] < 1), 'benigno',
                 'maligno'))

    df_clean = pd.DataFrame(path_input_csv[['path', 'binary_score']])
    return df_clean


def image_generator(dataframe,directory, target_size = (258, 258), batch_size = 64):
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
