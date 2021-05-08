<p align="right"><img src="https://cdn-images-1.medium.com/max/184/1*2GDcaeYIx_bQAZLxWM4PsQ@2x.png" width="80"></p>

# Prostate Biopsy Cancer Detector
**Owner**: Alvaro Gil (https://github.com/alvarogmf)
**Bootcamp**: Ironhack - Data Analytics Part Time Nov 2020

This is the final project for the Ironhack Bootcamp (Madrid, November 2020 - May 2021). It is a Image Classificator through a Convolutional Neural Network (CNN), able to identify if there is Cancer, or not, in Prostate Biopsies.

With more than 1 million new diagnoses reported every year, Prostate Cancer (PCa) is the second most common cancer among males worldwide that results in more than 350,000 deaths annually. Its early detection is capital for an adequate treatment and reduction of the death rate.

This Project pretends to be a simple web-app to be used by any doctor that wants to check if a biopsy has Cancer or not in just a few seconds, uploading the image in a .tiff format:

<p align="center">
<a href="https://ibb.co/rGC8P32"><img src="https://i.ibb.co/TBVjX1M/Captura.png" alt="Captura" border="0" width="350" height="400"></a>
</p>

## Overview

### :computer: **Technology stack**
Project made on Python 3. Main modules used:
 - Pandas
 - OpenCV
 - Tensorflow
 - Streamlit

### :mag_right: **Image Dataset**
The images used to develop this project and train the model come from the [Prostate cANcer graDe Assessment (PANDA) Challenge](https://www.kaggle.com/c/prostate-cancer-grade-assessment/overview)  in Kaggle. This dataset has more than **10k different biopsy images** that can be used to train the model. The total weight is almost **400Gb of data**, so it cannot be uploaded to GitHub. If you want to used this Project and use it to train, please download the images and save them in the train_images folder.
<p align="center">
<a href="https://ibb.co/dLhgQVC"><img src="https://i.ibb.co/qWZ5Rcq/00a7fb880dc12c5de82df39b30533da9.png" alt="00a7fb880dc12c5de82df39b30533da9" border="0"></a>
</p>

### :chart_with_upwards_trend: **Model**
The project has been developed using a custom Sequential model with the following layers:
|Layer (type) 	 |Output Shape					 |Param #					   |
|----------------|-------------------------------|-----------------------------|
|conv2d_3 (Conv2D) |(None, 254, 254, 16)            |448                   |
|max_pooling2d_3 (MaxPooling2)|(None, 127, 127, 16)         |0            |
|conv2d_4 (Conv2D)|(None, 125, 125, 32)|4640      |
|max_pooling2d_4 (MaxPooling2)|(None, 63, 63, 32)|0|
|dropout_5 (Dropout)|(None, 63, 63, 32)|0|
|conv2d_5 (Conv2D)| (None, 60, 60, 64) |18496     |
|max_pooling2d_5 (MaxPooling2)| (None, 30, 30, 64) |0|
|dropout_6 (Dropout)| (None, 30, 30, 64)  |0|
|flatten_1 (Flatten)| (None, 57600)    |0|
|dense_4 (Dense)|(None, 32)   |3686464|
|dropout_7 (Dropout)|(None, 64)   |0|
|dense_5 (Dense)|  (None, 32)   |2080|
|dropout_8 (Dropout)|(None, 32)   |0|
|dense_6 (Dense)|  (None, 16)   |528|
|dropout_9 (Dropout)|(None, 16) |0|
|dense_7 (Dense)|  (None, 1)   |17|
Total params: 3,712,673
Trainable params: 3,712,673
Non-trainable params: 0

### :video_game: **How to Use**
This project has two tools that can be used, one to be used in the Terminal (main.py) and the other to be used on the web browser (app.py).
#### main.py:
This script is a command based program, it requires certain arguments to work:
**Convert:**
If the commands `python main.py -convert` or `python main.py -c` are used, the program will take the TIFF images stored in the train_images folder and convert them into PNG format and resized them to 10% their original size. These new images will be stored in the png_images folder and are be the ones used to train the model.

**Train:**
If the commands `python main.py -train` or `python main.py -t` are used, it will train based on the parameters mentioned above and save the new trained model in the folder /model with the name model.h5 (it will rewrite the pre-existing model with the same name).
*NOTE_1: In case you want to train the model, it uses the PNG images created with the first argument (-convert). If there are no images in the png_images folder, please run first `python main.py -c`.
NOTE_2: This will take a long time to run, depending on the characteristics of the computer, we highly recommend not to do this unless is strictly necessary and in that case, use tensorflow-gpu to speed up the trainning.*

**Predict:**
If the commands `python main.py -predict` or `python main.py -p` are used, it wil create a prediction of the biopsy image (in TIFF format) stored in the predict folder. It will show on screen the prediction and create a PDF report that can be found in the results folder.
*NOTE: it predicts only ONE image at a time, please make sure there is only one TIFF image in the predict folder (don't worry about the png_converted folder, no need to do anything with that).*

#### app.py:
Although in future iterations I pretend to create a web-page that will store the app, for now it is locally stored. To access the Streamlit app, in the Terminal head to the folder and type `streamlit run app.py` this will open a local host page on your browser. Here you can upload a biopsy image (again, in TIFF format) and it will display an image of the biopsy and the prediction of whether if there is Cancer or not.

<p align="center">
<a href="https://ibb.co/fC5qBQp"><img src="https://i.ibb.co/rZSFTv6/streamlit-app-2021-05-05-19-05-32.gif" alt="streamlit-app-2021-05-05-19-05-32" border="0"></a>
</p>

### :shit: **ToDo**
 1. Improve model Accuracy
 2. Create webpage to be accesible by anyone
 3. Add "Download as PDF" button on Streamlit

### 💌  **Contact info**
If you have any question or want to contribute on this project, please don't hesitate to contact me! 
